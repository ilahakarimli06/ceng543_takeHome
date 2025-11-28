"""
Question 5 - Interpretability Analysis
Using the best model: Transformer + Random Embedding (4 layers, 2 heads)
Based on ablation study results from Question 3
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
from model_analyzer import ModelAnalyzer
import torch
import math
import math
import json
    
        

class InterpretabilityAnalyzer(ModelAnalyzer):
    """Interpretability analysis using attention visualization"""
    
    def visualize_attention(self, source_text, save_path=None):
        """
        Visualize attention weights as heatmap
        """
        translation, attention, src_tokens, tgt_tokens = self.translate_and_get_attention(source_text)
        src_words = [self.sp.id_to_piece(t) for t in src_tokens]
        tgt_words = [self.sp.id_to_piece(t) for t in tgt_tokens]
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention, 
                    xticklabels=src_words,
                    yticklabels=tgt_words,
                    cmap='Blues',
                    cbar_kws={'label': 'Attention Weight'})
        plt.xlabel('Source Tokens', fontsize=12)
        plt.ylabel('Target Tokens', fontsize=12)
        plt.title(f'Attention Weights\nSource: {source_text}\nTranslation: {translation}', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        # Save to output/visualizations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if save_path is None:
            save_path = os.path.join(script_dir, 'output', 'visualizations', 'attention_map.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Attention heatmap saved to {save_path}")
        print(f"Source: {source_text}")
        print(f"Translation: {translation}")
        return translation
    
    def analyze_token_importance(self, source_text):
        """
        Simple token importance analysis
        Which source tokens contribute most to translation
        """
        translation, attention, src_tokens, tgt_tokens = self.translate_and_get_attention(source_text)
        
        # Average attention across all target tokens
        avg_attention = attention.mean(axis=0)
        
        # Get token strings
        src_words = [self.sp.id_to_piece(t) for t in src_tokens]
        
        # Create importance ranking
        importance = list(zip(src_words, avg_attention))
        importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n=== Token Importance Analysis ===")
        print(f"Source: {source_text}")
        print(f"Translation: {translation}")
        print("\nMost important source tokens:")
        for i, (token, score) in enumerate(importance[:5], 1):
            print(f"{i}. '{token}' - importance: {score:.4f}")
        
        return importance
    
    def integrated_gradients(self, source_text, n_steps=50):
        """
        Integrated Gradients for token importance.
        Uses zero embedding as baseline and linearly interpolates n_steps.
        """

        self.model.train()  # enable grads
        
        # Tokenize
        src_tokens = self.tokenize(source_text)
        src_tensor = torch.tensor([src_tokens]).to(self.device)
        
        pad_id = self.model.pad_id if hasattr(self.model, 'pad_id') else 0
        src_key_padding_mask = (src_tensor == pad_id)
        tgt_tensor = torch.tensor([[self.sp.bos_id()]]).to(self.device)
        tgt_key_padding_mask = (tgt_tensor == pad_id)
        tgt_mask = self.model.generate_square_subsequent_mask(1).to(self.device)

        # Baseline (zeros) and actual embedding
        base_emb = torch.zeros_like(self.model.source_embedding(src_tensor))
        src_emb = self.model.source_embedding(src_tensor)

        total_grad = torch.zeros_like(src_emb)

        # Line integral approximation
        for step in range(n_steps):
            alpha = float(step + 1) / n_steps
            interpolated = base_emb + alpha * (src_emb - base_emb)
            interpolated.requires_grad_(True)
            interpolated = interpolated * math.sqrt(self.model.model_dim)
            interpolated = self.model.positional_encoding(interpolated)
            interpolated.retain_grad()

            tgt_emb = self.model.target_embedding(tgt_tensor) * math.sqrt(self.model.model_dim)
            tgt_emb = self.model.positional_encoding(tgt_emb)

            output = self.model.transformer(
                interpolated,
                tgt_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            logits = self.model.output_projection(output)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top_prob = probs.max()

            # Backprop through this scaled input (single use graph)
            self.model.zero_grad()
            grad = torch.autograd.grad(
                outputs=top_prob,
                inputs=interpolated,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )[0]

            if grad is not None:
                total_grad += grad.detach()

        # IG: average gradient along path * (input - baseline)
        avg_grad = total_grad / max(1, n_steps)
        ig = (src_emb - base_emb) * avg_grad
        ig_scores = ig.sum(dim=-1).squeeze().abs().detach().cpu().numpy()
        self.model.eval()
        
        # Get token strings
        src_words = [self.sp.id_to_piece(t) for t in src_tokens]
        
        # Create importance ranking
        importance = list(zip(src_words, ig_scores))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        self.model.eval()  # Back to eval mode
        
        return importance
    
    def visualize_integrated_gradients(self, source_text, save_path=None):
        """
        Visualize Integrated Gradients token importance
        """
        print(f"\nComputing Integrated Gradients for: {source_text}")
        
        # Compute IG
        importance = self.integrated_gradients(source_text)
        
        # Separate tokens and scores
        tokens = [item[0] for item in importance]
        scores = [item[1] for item in importance]
        
        # Plot bar chart
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        colors = ['green' if s > 0 else 'red' for s in scores]
        plt.barh(range(len(tokens)), scores, color=colors, alpha=0.6)
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel('Integrated Gradients Score', fontsize=12)
        plt.ylabel('Source Tokens', fontsize=12)
        plt.title(f'Integrated Gradients - Token Importance\nSource: {source_text}', fontsize=10)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        # Save to output/visualizations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if save_path is None:
            save_path = os.path.join(script_dir, 'output', 'visualizations', 'ig_importance.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"IG visualization saved to {save_path}")
        print("\nTop 5 most important tokens (by absolute importance):")
        for i, (token, score) in enumerate(importance[:5], 1):
            print(f"{i}. '{token}' - IG score: {score:.4f}")
        return importance


def run_interpretability_analysis(analyzer):
    """Run interpretability analysis with existing analyzer"""

    # Load model info
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
            best_val_loss = history.get('best_val_loss', 0)
            best_ppl = math.exp(best_val_loss) if best_val_loss > 0 else 0
    except:
        best_ppl = 0
    
    # Print model info
    config = analyzer.config
    num_layers = config.get('num_layers', '?')
    nhead = config.get('num_heads', '?')
    embedding_type = config.get('embedding_type', 'random')
    
    # Test sentences (German to English)
    test_sentences = [
        "Ich liebe maschinelles Lernen.",
        "Der Hund spielt im Garten.",
        "Die Katze schläft auf dem Sofa.",
        "Das Wetter ist heute sehr schön.",
        "Wir gehen morgen ins Kino."
    ]
    
    print("="*60)
    print("INTERPRETABILITY ANALYSIS")
    print(f"Model: Transformer + {embedding_type.title()} ({num_layers} layers, {nhead} heads)")
    if best_ppl > 0:
        print(f"Perplexity: {best_ppl:.2f}")
    print("="*60)
    
    # Visualize attention for each sentence
    print("\n" + "="*60)
    print("PART 1: ATTENTION VISUALIZATION")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Example {i} ---")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'output', 'visualizations', f'attention_map_{i}.png')
        translation = analyzer.visualize_attention(sentence, save_path)
        analyzer.analyze_token_importance(sentence)
        print()
    print("\n" + "="*60)
    print("PART 2: INTEGRATED GRADIENTS")
    print("="*60)
    print("(Computing gradient-based importance for 3 examples...)\n")
    for i, sentence in enumerate(test_sentences[:3], 1):
        print(f"\n--- Example {i} ---")
        save_path = os.path.join(script_dir, 'output', 'visualizations', f'ig_importance_{i}.png')
        analyzer.visualize_integrated_gradients(sentence, save_path)
        print()
    print("\n Interpretability analysis complete!")
    print("  - Attention maps: output/visualizations/attention_map_*.png (5 examples)")
    print("  - Integrated Gradients: output/visualizations/ig_importance_*.png (3 examples)")
    return analyzer
