"""
Question 3b & 3c: Train Transformer model with different embeddings
Trains with: Random, GloVe, and BERT embeddings
Based on "Attention Is All You Need" (Vaswani et al., 2017)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import math
import time
import json
import sentencepiece as spm
from torch.utils.data import DataLoader

from question2.utils.dataset import ParallelIdsDataset, collate_pad, PAD_ID
from question3.models.transformer import TransformerModel
from question3.utils.embeddings import GloVeEmbeddings, create_embedding_layer
from question3.evaluate import evaluate_bleu_rouge


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
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
        
        # Forward pass
        logits, _ = model(batch)
        
        # Compute loss
        loss = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')(
            logits.view(-1, logits.size(-1)), 
            batch["tgt_out"].view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_tokens += (batch["tgt_out"] != PAD_ID).sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))
    return perplexity


@torch.no_grad()
def evaluate(model, loader, device):
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
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))
    return perplexity


def train_with_embedding(embedding_type, device, epochs=10):
    """Train Transformer with specific embedding type"""
    set_seed(42)  # For reproducibility
    start_time = time.time()
    
    # Parameters (consistent with Question 2)
    VOCAB_SIZE = 8000
    NUM_LAYERS = 2
    DIM_FFN = 512
    DROPOUT = 0.1
    BATCH_SIZE = 64
    LR = 3e-4
    SPM_MODEL = "question2/data/spm_shared_unigram.model"
    
    # All use same tokenized data
    TRAIN_SRC = "question2/data/tokenized/train.en.ids"
    TRAIN_TGT = "question2/data/tokenized/train.de.ids"
    VALID_SRC = "question2/data/tokenized/validation.en.ids"
    VALID_TGT = "question2/data/tokenized/validation.de.ids"
    SAVE_PATH = f"question3/transformer_{embedding_type}_model.pt"
    
    print(f"\n{'='*70}")
    print(f"Training Transformer with {embedding_type.upper()} embeddings")
    print(f"{'='*70}\n")
    
    # Create embeddings aligned to SentencePiece vocabulary
    if embedding_type == "random":
        D_MODEL = 256
        NHEAD = 8  
        source_emb = create_embedding_layer(VOCAB_SIZE, D_MODEL, "random", PAD_ID)
        target_emb = create_embedding_layer(VOCAB_SIZE, D_MODEL, "random", PAD_ID)
    elif embedding_type == "glove":
        print("Loading GloVe and aligning to SentencePiece vocabulary...")
        glove = GloVeEmbeddings(name="6B", dim=300)
        source_emb = create_embedding_layer(VOCAB_SIZE, 300, "glove", PAD_ID, SPM_MODEL, glove)
        target_emb = create_embedding_layer(VOCAB_SIZE, 300, "glove", PAD_ID, SPM_MODEL, glove)
        D_MODEL = 300
        NHEAD = 10  # when it was 8, I got an error since 300 is not divisible by 8
    elif embedding_type == "bert":
        print("Creating BERT-aligned embeddings for SentencePiece vocabulary...")
        source_emb = create_embedding_layer(VOCAB_SIZE, 768, "bert", PAD_ID, SPM_MODEL)
        target_emb = create_embedding_layer(VOCAB_SIZE, 768, "bert", PAD_ID, SPM_MODEL)
        D_MODEL = 768
        NHEAD = 8  
    
    # Data loaders (same for all embedding types)
    train_loader = DataLoader(
        ParallelIdsDataset(TRAIN_SRC, TRAIN_TGT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    
    valid_loader = DataLoader(
        ParallelIdsDataset(VALID_SRC, VALID_TGT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    
    # Model
    model = TransformerModel(
        source_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        model_dim=D_MODEL,
        num_heads=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        feedforward_dim=DIM_FFN,
        dropout=DROPOUT,
        pad_id=PAD_ID,
        source_embedding=source_emb,
        target_embedding=target_emb
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: {NUM_LAYERS} layers, {NHEAD} attention heads, model_dim={D_MODEL}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    best_ppl = float('inf')
    train_losses = []
    epoch_times = []
    
    print(f"\nTraining started...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_ppl = train_one_epoch(model, train_loader, optimizer, device)
        valid_ppl = evaluate(model, valid_loader, device)
        epoch_time = time.time() - epoch_start
        
        epoch_times.append(epoch_time)
        train_losses.append(train_ppl)
        
        # GPU memory
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        
        print(f"Epoch {epoch:02d} | Train Perplexity: {train_ppl:.2f} | Valid Perplexity: {valid_ppl:.2f} | Time: {epoch_time:.1f}s | GPU: {gpu_mem:.2f}GB")
        
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_perplexity': valid_ppl,
                'embedding_type': embedding_type
            }, SAVE_PATH)
            print(f"  → Saved best model (Perplexity: {best_ppl:.2f})")
    
    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print(f"\nCompleted. Best validation perplexity: {best_ppl:.2f}")
    print(f"Total training time: {total_time:.1f}s | Avg per epoch: {avg_epoch_time:.1f}s")
    
    # Evaluate BLEU and ROUGE on test set
    print(f"\nEvaluating BLEU and ROUGE on test set...")
    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
    TEST_SRC = "question2/data/tokenized/test.en.ids"
    TEST_TGT = "question2/data/tokenized/test.de.ids"
    bleu_score, rouge_scores = evaluate_bleu_rouge(model, TEST_SRC, TEST_TGT, sp, device, max_length=50, is_transformer=True)
    
    print(f"BLEU: {bleu_score:.2f}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f} | ROUGE-2: {rouge_scores['rouge2']:.4f} | ROUGE-L: {rouge_scores['rougeLsum']:.4f}")
    
    return {
        'best_ppl': best_ppl,
        'bleu': bleu_score,
        'rouge': rouge_scores,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'train_losses': train_losses
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    results = {}
    
    # Train with all embedding types
    for embedding_type in ["random", "glove", "bert"]:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        results[embedding_type] = train_with_embedding(embedding_type, device, epochs=20)
    
    # Print comparison
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS - Transformer with Different Embeddings")
    print(f"{'='*80}\n")
    print(f"{'Embedding':<15} {'Perplexity':>12} {'BLEU':>8} {'ROUGE-1':>10} {'Time(s)':>10} {'GPU(GB)':>10}")
    print("-" * 80)
    
    for embedding_type, results_dict in results.items():
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"{embedding_type:<15} {results_dict['best_ppl']:>12.2f} {results_dict['bleu']:>8.2f} {results_dict['rouge']['rouge1']:>10.4f} {results_dict['total_time']:>10.1f} {gpu_memory:>10.2f}")
    
    print(f"\n{'='*80}\n")
    
    # Save results to JSON
    import os
    os.makedirs("question3/output", exist_ok=True)
    output_file = "question3/output/transformer_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
