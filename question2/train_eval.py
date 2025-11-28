import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm
import sacrebleu
from rouge_score import rouge_scorer

from utils.dataset import ParallelIdsDataset, collate_pad, PAD_ID, read_ids
from models.seq2seq import Seq2Seq


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(batch)
        
        loss = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')(
            logits.view(-1, logits.size(-1)), 
            batch["tgt_out"].view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_tokens += (batch["tgt_out"] != PAD_ID).sum().item()
    
    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    # Clip to prevent overflow
    perplexity = math.exp(min(avg_loss, 100))
    return perplexity


@torch.no_grad()
def evaluate_perplexity(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(batch)
        
        loss = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')(
            logits.view(-1, logits.size(-1)), 
            batch["tgt_out"].view(-1)
        )
        
        total_loss += loss.item()
        total_tokens += (batch["tgt_out"] != PAD_ID).sum().item()
    
    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    # Clip to prevent overflow
    perplexity = math.exp(min(avg_loss, 100))
    return perplexity


@torch.no_grad()
def greedy_decode(model, src_ids, sp, max_len, device):
    model.eval()
    bos_id, eos_id = sp.bos_id(), sp.eos_id()
    
    src = torch.tensor([src_ids], device=device, dtype=torch.long)
    source_mask = (src == PAD_ID)
    enc_outs, (h, c) = model.encoder(src, source_mask)
    
    tokens = []
    prev_token = torch.tensor([bos_id], device=device, dtype=torch.long)
    
    for _ in range(max_len):
        emb = model.decoder.drop(model.decoder.emb(prev_token)).unsqueeze(1)
        context, _ = model.decoder.attn(h[-1], enc_outs, source_mask)
        context = context.unsqueeze(1)
        
        rnn_in = torch.cat([emb, context], dim=-1)
        out, (h, c) = model.decoder.rnn(rnn_in, (h, c))
        
        logits = model.decoder.fc_out(torch.cat([out, context, emb], dim=-1))
        next_token = logits.argmax(-1).squeeze()
        
        tokens.append(next_token.item())
        if next_token.item() == eos_id:
            break
        prev_token = next_token.unsqueeze(0)
    
    return tokens


def evaluate_bleu_rouge(model, src_path, tgt_path, sp, device, max_len=50):
    model.eval()
    src_seqs = read_ids(src_path)
    tgt_seqs = read_ids(tgt_path)
    
    hypotheses, references = [], []
    
    for src_ids, tgt_ids in zip(src_seqs, tgt_seqs):
        pred_ids = greedy_decode(model, src_ids, sp, max_len, device)
        hypotheses.append(sp.decode(pred_ids))
        references.append(sp.decode(tgt_ids))
    
    # BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}
    

    #Edited with the help of AI
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeLsum'] += scores['rougeLsum'].fmeasure
    
    n = len(hypotheses)
    rouge_scores = {k: v/n for k, v in rouge_scores.items()}
    
    return bleu.score, rouge_scores


@torch.no_grad()
def compute_attention_metrics(model, loader, device):
    """Compute average entropy and sharpness across dataset"""
    model.eval()
    total_entropy, total_sharpness, total_steps = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, attn = model(batch) 
        
        # Mask padding in source
        source_mask = batch['source_mask'] 
        attn_masked = attn.clone()
        attn_masked[source_mask.unsqueeze(1).expand_as(attn)] = 0.0
        
        # Normalize attention
        attn_normalized = attn_masked / (attn_masked.sum(-1, keepdim=True) + 1e-9)
        
        # Entropy: -sum(p * log(p))
        entropy = -(attn_normalized * (attn_normalized + 1e-12).log()).sum(-1)  
        
        # Sharpness: max attention weight per step
        sharpness = attn_normalized.max(dim=-1).values 

        # Mask padding in target
        target_mask = batch['target_mask'] 
        valid_steps = ~target_mask
        
        total_entropy += (entropy * valid_steps).sum().item()
        total_sharpness += (sharpness * valid_steps).sum().item()
        total_steps += valid_steps.sum().item()
    
    avg_entropy = total_entropy / total_steps
    avg_sharpness = total_sharpness / total_steps
    
    return avg_entropy, avg_sharpness


def train(train_src, train_tgt, valid_src, valid_tgt, 
          vocab_size, attn_type, save_path, 
          epochs=10, batch_size=64, lr=3e-4, device='cuda'):
    
    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(
        ParallelIdsDataset(train_src, train_tgt),
        batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    
    valid_loader = DataLoader(
        ParallelIdsDataset(valid_src, valid_tgt),
        batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    
    model = Seq2Seq(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        emb_dim=256,
        enc_hid=256,
        dec_hid=256,
        num_layers=2,
        attn_type=attn_type,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_perplexity = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_perplexity = train_one_epoch(model, train_loader, optimizer, device)
        valid_perplexity = evaluate_perplexity(model, valid_loader, device)
        
        print(f"Epoch {epoch:02d} | Train Perplexity: {train_perplexity:.2f} | Valid Perplexity: {valid_perplexity:.2f}")
        
        if valid_perplexity < best_perplexity:
            best_perplexity = valid_perplexity
            torch.save({
                'model': model.state_dict(),
                'cfg': {
                    'src_vocab': vocab_size,
                    'tgt_vocab': vocab_size,
                    'embedding_dim': 256,
                    'encoder_hidden': 256,
                    'decoder_hidden': 256,
                    'attention_type': attn_type,
                    'num_layers': 2,
                    'dropout': 0.1
                }
            }, save_path)
            print(f"  → Saved best model (Perplexity: {best_perplexity:.2f}")
    
    return save_path
