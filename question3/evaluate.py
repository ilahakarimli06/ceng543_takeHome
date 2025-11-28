"""
Evaluation utilities for Question 3
BLEU, ROUGE, and translation generation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import sacrebleu
from rouge_score import rouge_scorer
from question2.utils.dataset import read_ids


def read_text_lines(path):
    """Read text lines from a file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def greedy_decode_seq2seq(model, source_ids, sentence_piece, max_length, device):
    """Greedy decoding for Seq2Seq model."""
    model.eval()
    begin_of_sequence_id = sentence_piece.bos_id()
    end_of_sequence_id = sentence_piece.eos_id()
    PAD_ID = 0
    
    source = torch.tensor([source_ids], device=device)
    source_mask = (source == PAD_ID)
    encoder_outputs, (hidden, cell) = model.encoder(source, source_mask)
    
    tokens = []
    previous_token = torch.tensor([begin_of_sequence_id], device=device)
    
    for _ in range(max_length):
        embedded = model.decoder.embedding(previous_token)
        embedded = model.decoder.dropout(embedded).unsqueeze(1)
        context, _ = model.decoder.attention(hidden[-1], encoder_outputs, source_mask)
        context = context.unsqueeze(1)
        
        rnn_input = torch.cat([embedded, context], dim=-1)
        output, (hidden, cell) = model.decoder.rnn(rnn_input, (hidden, cell))
        
        logits = model.decoder.output_projection(torch.cat([output, context, embedded], dim=-1))
        next_token = logits.argmax(-1).squeeze()
        
        tokens.append(next_token.item())
        if next_token.item() == end_of_sequence_id:
            break
        
        previous_token = next_token.unsqueeze(0)
    
    return tokens


@torch.no_grad()
def greedy_decode_transformer(model, source_ids, sentence_piece, max_length, device):
    """Greedy decoding for Transformer model."""
    model.eval()

    # SentencePiece special tokens
    begin_of_sequence_id = sentence_piece.bos_id()
    end_of_sequence_id = sentence_piece.eos_id()
    PAD_ID = 0
    
     # Prepare source tensor (batch size = 1)
    source = torch.tensor([source_ids], device=device)
    source_mask = (source == PAD_ID)
    

    # Initialize target sequence with BOS token
    target = torch.tensor([[begin_of_sequence_id]], device=device)
    

    # Greedy decoding loop
    for _ in range(max_length):

        # Create target attention mask (no masking except padding)
        target_mask = torch.zeros(1, target.size(1), dtype=torch.bool, device=device)
        

         # Pack batch dictionary expected by the model
        batch = {
            "src": source,
            "target_input": target,
            "source_mask": source_mask,
            "target_mask": target_mask
        }
        
        # Forward pass through the model
        logits, _ = model(batch)
        next_token = logits[0, -1].argmax().item()
        
        if next_token == end_of_sequence_id:
            break
        
        target = torch.cat([target, torch.tensor([[next_token]], device=device)], dim=1)
    
    return target[0, 1:].tolist()


@torch.no_grad()
def greedy_decode_transformer_with_attention(model, source_ids, sentence_piece, max_length, device):
    """
    Greedy decode and return encoder-decoder attention from the final decoding step.
    Attention is averaged over heads if MultiheadAttention returns a batch dimension.
    """
    model.eval()
    bos_id = sentence_piece.bos_id()
    eos_id = sentence_piece.eos_id()
    pad_id = 0

    source = torch.tensor([source_ids], device=device)
    source_mask = (source == pad_id)
    target = torch.tensor([[bos_id]], device=device)

    last_attention = None

    for _ in range(max_length):
        target_mask = torch.zeros(1, target.size(1), dtype=torch.bool, device=device)
        batch = {
            "src": source,
            "target_input": target,
            "source_mask": source_mask,
            "target_mask": target_mask
        }
        logits, attentions = model(batch)

        # Save attention from the last decoder layer (most informative)
        if attentions:
            last_attention = attentions[-1]

        next_token = logits[0, -1].argmax().item()
        if next_token == eos_id:
            break

        target = torch.cat([target, torch.tensor([[next_token]], device=device)], dim=1)

    predicted_ids = target[0, 1:].tolist()

    # Normalize attention to shape (tgt_len, src_len)
    attn_map = None
    if last_attention is not None:
        attn = last_attention
        # Possible shapes: (batch, tgt, src), (num_heads, tgt, src), (batch, heads, tgt, src), or (tgt, src)
        if attn.dim() == 4:  # (batch, heads, tgt, src)
            attn = attn.mean(dim=1)
            attn = attn[0] if attn.size(0) == 1 else attn.mean(dim=0)
        elif attn.dim() == 3:
            attn = attn[0] if attn.size(0) == 1 else attn.mean(dim=0)
        attn_map = attn
        # Drop BOS row to align with predicted_ids length
        if attn_map is not None and attn_map.size(0) == len(predicted_ids) + 1:
            attn_map = attn_map[1:]

    return predicted_ids, attn_map


def evaluate_bleu_rouge(model, source_path, target_path, sentence_piece, device, max_length=50, is_transformer=False):
    """Evaluate BLEU and ROUGE scores for machine translation."""
    model.eval()
    
    # Read tokenized IDs (all embedding types use SentencePiece)
    source_sequences = read_ids(source_path)
    target_sequences = read_ids(target_path)
    
    hypotheses, references = [], []
    
    print(f"Translating {len(source_sequences)} sentences...")
    for index, (source_ids, target_ids) in enumerate(zip(source_sequences, target_sequences)):
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(source_sequences)} completed")
        
        if is_transformer:
            predicted_ids = greedy_decode_transformer(model, source_ids, sentence_piece, max_length, device)
        else:
            predicted_ids = greedy_decode_seq2seq(model, source_ids, sentence_piece, max_length, device)
        
        hypotheses.append(sentence_piece.decode(predicted_ids))
        references.append(sentence_piece.decode(target_ids))
    
    # BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}
    
    for hypothesis, reference in zip(hypotheses, references):
        scores = scorer.score(reference, hypothesis)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeLsum'] += scores['rougeLsum'].fmeasure
    
    num_samples = len(hypotheses)
    rouge_scores = {key: value / num_samples for key, value in rouge_scores.items()}
    
    return bleu.score, rouge_scores


# --- BERT-tokenizer variant (without SentencePiece) ---------------------------------

@torch.no_grad()
def greedy_decode_seq2seq_bert(model, source_ids, tokenizer, max_length, device):
    """Greedy decoding for Seq2Seq model when data is tokenized with BERT tokenizer."""
    model.eval()
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0
    
    source = torch.tensor([source_ids], device=device)
    source_mask = (source == pad_id)
    encoder_outputs, (hidden, cell) = model.encoder(source, source_mask)
    
    tokens = []
    previous_token = torch.tensor([bos_id], device=device)
    
    for _ in range(max_length):
        embedded = model.decoder.embedding(previous_token)
        embedded = model.decoder.dropout(embedded).unsqueeze(1)
        context, _ = model.decoder.attention(hidden[-1], encoder_outputs, source_mask)
        context = context.unsqueeze(1)
        
        rnn_input = torch.cat([embedded, context], dim=-1)
        output, (hidden, cell) = model.decoder.rnn(rnn_input, (hidden, cell))
        
        logits = model.decoder.output_projection(torch.cat([output, context, embedded], dim=-1))
        next_token = logits.argmax(-1).squeeze()
        
        tokens.append(next_token.item())
        if next_token.item() == eos_id:
            break
        
        previous_token = next_token.unsqueeze(0)
    
    return tokens


@torch.no_grad()
def evaluate_bleu_rouge_bert(model, source_path, target_path, tokenizer, device, max_length=50):
    """Evaluate BLEU/ROUGE when inputs are tokenized with a BERT tokenizer."""
    model.eval()
    
    # Read pre-tokenized input sequences (list of ID lists)
    source_sequences = read_ids(source_path)
    target_sequences = read_ids(target_path)
    
    hypotheses, references = [], []
    
    print(f"Translating {len(source_sequences)} sentences...")
    for index, (source_ids, target_ids) in enumerate(zip(source_sequences, target_sequences)):
        
        # Progress update every 100 samples
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(source_sequences)} completed")
        
        
        predicted_ids = greedy_decode_seq2seq_bert(model, source_ids, tokenizer, max_length, device)
        
        # Convert predicted IDs and gold target IDs back to text
        hypotheses.append(tokenizer.decode(predicted_ids, skip_special_tokens=True))
        references.append(tokenizer.decode(target_ids, skip_special_tokens=True))
    
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}
    
    for hypothesis, reference in zip(hypotheses, references):
        scores = scorer.score(reference, hypothesis)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeLsum'] += scores['rougeLsum'].fmeasure
    
    # Average ROUGE over all samples
    num_samples = len(hypotheses)
    rouge_scores = {key: value / num_samples for key, value in rouge_scores.items()}
    
    return bleu.score, rouge_scores
