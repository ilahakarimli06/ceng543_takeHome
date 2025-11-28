"""
Question 3e: Ablation study - varying layers and attention heads
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import json
import math
import time
import sentencepiece as spm
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from question2.utils.dataset import ParallelIdsDataset, collate_pad, PAD_ID
from question3.models.transformer import TransformerModel
from question3.evaluate import evaluate_bleu_rouge


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_configuration(num_layers, num_heads, device):
    set_seed(42)  # For reproducibility
    start_time = time.time()
    
    train_loader = DataLoader(
        ParallelIdsDataset("question2/data/tokenized/train.en.ids", "question2/data/tokenized/train.de.ids"),
        batch_size=64, shuffle=True, collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    validation_loader = DataLoader(
        ParallelIdsDataset("question2/data/tokenized/validation.en.ids", "question2/data/tokenized/validation.de.ids"),
        batch_size=64, shuffle=False, collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    
    model = TransformerModel(
        source_vocab_size=8000, 
        target_vocab_size=8000, 
        model_dim=256, 
        num_heads=num_heads, 
        num_encoder_layers=num_layers, 
        num_decoder_layers=num_layers, 
        feedforward_dim=512, 
        dropout=0.1, 
        pad_id=PAD_ID
    ).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    
    best_perplexity = float('inf')
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, _ = model(batch)
            loss = CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')(
                logits.view(-1, logits.size(-1)), batch["tgt_out"].view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, _ = model(batch)
                validation_loss += CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')(
                    logits.view(-1, logits.size(-1)), batch["tgt_out"].view(-1)
                ).item()
        
        perplexity = math.exp(min(validation_loss / sum((b["tgt_out"] != PAD_ID).sum().item() for b in validation_loader), 100))
        print(f"Epoch {epoch:02d} | Validation Perplexity: {perplexity:.2f}")
        if perplexity < best_perplexity:
            best_perplexity = perplexity
    
    sentence_piece = spm.SentencePieceProcessor(model_file="question2/data/spm_shared_unigram.model")
    bleu_score, rouge_scores = evaluate_bleu_rouge(
        model, "question2/data/tokenized/test.en.ids", "question2/data/tokenized/test.de.ids", 
        sentence_piece, device, max_length=50, is_transformer=True
    )
    
    return {
        'perplexity': best_perplexity, 
        'bleu': bleu_score, 
        'rouge1': rouge_scores['rouge1'], 
        'time': time.time() - start_time
    }


def main():
    """Run ablation study varying number of layers and attention heads."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    results = {}
    
    # Full grid search: all combinations of layers and heads
    layer_configs = [1, 2, 4]
    head_configs = [2, 4, 8]
    
    print("="*70)
    print("ABLATION STUDY: Full Grid Search (Layers × Heads)")
    print("="*70)
    
    for num_layers in layer_configs:
        for num_heads in head_configs:
            config_name = f"layers_{num_layers}_heads_{num_heads}"
            print(f"\n{num_layers} layers, {num_heads} attention heads")
            results[config_name] = train_configuration(num_layers, num_heads, device)
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    for configuration_name, result in results.items():
        print(f"{configuration_name}: BLEU={result['bleu']:.2f}, Perplexity={result['perplexity']:.2f}, Time={result['time']:.1f}s")
    
    import os
    os.makedirs("question3/output", exist_ok=True)
    with open("question3/output/ablation_results.json", "w") as file:
        json.dump(results, file, indent=4)
    print(f"\nResults saved to: question3/output/ablation_results.json")


if __name__ == "__main__":
    main()
