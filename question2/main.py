"""
Train all three attention mechanisms and evaluate with BLEU, ROUGE, and Perplexity.
"""
import torch
import json
import sentencepiece as spm
from train_eval import train, evaluate_bleu_rouge, evaluate_perplexity, compute_attention_metrics
from models.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from utils.dataset import ParallelIdsDataset, collate_pad, PAD_ID
from viz import visualize_attention

VOCAB_SIZE = 8000
DATA_DIR = "question2/data/tokenized"
OUTPUT_DIR = "question2/output"
SPM_MODEL = "question2/data/spm_shared_unigram.model"

ATTENTION_TYPES = ["additive", "multiplicative", "scaled_dot"]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create output directories
    from pathlib import Path
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/attn_maps").mkdir(parents=True, exist_ok=True)
    
    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
    results = {}
    
    # Train each attention type
    for attn_type in ATTENTION_TYPES:
        print(f"\n{'='*70}")
        print(f"Training {attn_type.upper()} attention")
        print(f"{'='*70}\n")
        
        save_path = f"{OUTPUT_DIR}/{attn_type}_model.pt"
        
        train(
            train_src=f"{DATA_DIR}/train.en.ids",
            train_tgt=f"{DATA_DIR}/train.de.ids",
            valid_src=f"{DATA_DIR}/validation.en.ids",
            valid_tgt=f"{DATA_DIR}/validation.de.ids",
            vocab_size=VOCAB_SIZE,
            attn_type=attn_type,
            save_path=save_path,
            epochs=20,
            batch_size=64,
            lr=3e-4,
            device=device
        )
        
        print(f"\n{'='*70}")
        print(f"Evaluating {attn_type.upper()} on test set")
        print(f"{'='*70}\n")
        
        # Load best model
        checkpoint = torch.load(save_path, map_location=device)
        config = checkpoint['cfg']
        model = Seq2Seq(
            src_vocab_size=config.get('src_vocab', VOCAB_SIZE),
            tgt_vocab_size=config.get('tgt_vocab', VOCAB_SIZE),
            emb_dim=config.get('embedding_dim', config.get('emb_dim', 256)),
            enc_hid=config.get('encoder_hidden', config.get('enc_hid', 256)),
            dec_hid=config.get('decoder_hidden', config.get('dec_hid', 256)),
            num_layers=config.get('num_layers', 2),
            attn_type=config.get('attention_type', config.get('attn_type', attn_type)),
            dropout=config.get('dropout', 0.1)
        ).to(device)
        model.load_state_dict(checkpoint['model'])
        
        # Evaluate perplexity
        test_loader = DataLoader(
            ParallelIdsDataset(f"{DATA_DIR}/test.en.ids", f"{DATA_DIR}/test.de.ids"),
            batch_size=64, shuffle=False,
            collate_fn=lambda b: collate_pad(b, PAD_ID)
        )
        test_perplexity = evaluate_perplexity(model, test_loader, device)
        
        # Evaluate BLEU and ROUGE
        bleu_score, rouge_scores = evaluate_bleu_rouge(
            model, 
            f"{DATA_DIR}/test.en.ids",
            f"{DATA_DIR}/test.de.ids",
            sp, device, max_len=50
        )
        
        # Compute attention metrics
        entropy, sharpness = compute_attention_metrics(model, test_loader, device)
        
        results[attn_type] = {
            'BLEU': bleu_score,
            'ROUGE-1': rouge_scores['rouge1'],
            'ROUGE-2': rouge_scores['rouge2'],
            'ROUGE-L': rouge_scores['rougeLsum'],
            'Perplexity': test_perplexity,
            'Entropy': entropy,
            'Sharpness': sharpness
        }
        
        print(f"\nResults for {attn_type}:")
        print(f"  BLEU:       {bleu_score:.2f}")
        print(f"  ROUGE-1:    {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2:    {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L:    {rouge_scores['rougeLsum']:.4f}")
        print(f"  Perplexity: {test_perplexity:.2f}")
        print(f"  Entropy:    {entropy:.4f}")
        print(f"  Sharpness:  {sharpness:.4f}")
        
        # Visualize attention maps
        print(f"\nGenerating attention maps for {attn_type}...")
        visualize_attention(save_path, f"{OUTPUT_DIR}/attn_maps/{attn_type}")
    
    # Final comparison
    print(f"\n\n{'='*80}")
    print("FINAL COMPARISON - TASK (d) & (e)")
    print(f"{'='*80}\n")
    print(f"{'Attention':<18} {'BLEU':>8} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} {'PPL':>8} {'Entropy':>10} {'Sharpness':>10}")
    print("-" * 80)
    for attn_type in ATTENTION_TYPES:
        r = results[attn_type]
        print(f"{attn_type:<18} {r['BLEU']:>8.2f} {r['ROUGE-1']:>10.4f} {r['ROUGE-2']:>10.4f} {r['ROUGE-L']:>10.4f} {r['Perplexity']:>8.2f} {r['Entropy']:>10.4f} {r['Sharpness']:>10.4f}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS - TASK (e)")
    print(f"{'='*80}\n")
    
    # Sort by performance
    sorted_by_bleu = sorted(results.items(), key=lambda x: x[1]['BLEU'], reverse=True)
    best_attn = sorted_by_bleu[0][0]
    
    print(f"Best performing attention: {best_attn.upper()}")
    print(f"  BLEU: {results[best_attn]['BLEU']:.2f}")
    print(f"  Entropy: {results[best_attn]['Entropy']:.4f}")
    print(f"  Sharpness: {results[best_attn]['Sharpness']:.4f}")
    
    print(f"\nAttention Sharpness & Entropy Analysis:")
    print(f"  - Higher sharpness = more focused attention on specific source tokens")
    print(f"  - Lower entropy = less uniform distribution, more selective")
    print(f"  - Higher sharpness + Lower entropy → Better translation quality\n")
    
    for attn_type in ATTENTION_TYPES:
        r = results[attn_type]
        selectivity = "HIGH" if r['Sharpness'] > 0.5 else "MEDIUM" if r['Sharpness'] > 0.3 else "LOW"
        print(f"  {attn_type:>15}: Sharpness={r['Sharpness']:.4f}, Entropy={r['Entropy']:.4f} → Selectivity={selectivity}")
    print()
    
    # Save results to JSON
    output_file = f"{OUTPUT_DIR}/attention_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
