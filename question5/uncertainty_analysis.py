"""
Question 5 (d) - Uncertainty Quantification
Quantify model uncertainty using entropy and calibration metrics
"""
import torch
import numpy as np
from model_analyzer import ModelAnalyzer
import json
import os


class UncertaintyAnalyzer(ModelAnalyzer):
    """Analyze model uncertainty in predictions"""
    
    def calculate_token_entropy(self, logits):

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return entropy.item()
    
    def translate_with_uncertainty(self, source_text, max_len=50):
        """Translate with token-level uncertainty tracking"""
        self.model.eval()
        
        # Tokenize source
        src_tokens = self.tokenize(source_text)
        src_tensor = torch.tensor([src_tokens]).to(self.device)
        
        # Decode with uncertainty tracking
        tgt_tokens = [self.sp.bos_id()]
        token_entropies = []
        token_confidences = []
        
        for i in range(max_len):
            tgt_tensor = torch.tensor([tgt_tokens]).to(self.device)
            
            # Create masks
            pad_id = self.model.pad_id if hasattr(self.model, 'pad_id') else 0
            src_key_padding_mask = (src_tensor == pad_id)
            tgt_key_padding_mask = (tgt_tensor == pad_id)
            
            # Prepare batch
            batch = {
                'src': src_tensor,
                'target_input': tgt_tensor,
                'source_mask': src_key_padding_mask,
                'target_mask': tgt_key_padding_mask
            }
            
            with torch.no_grad():
                logits, _ = self.model(batch)
                logits_last = logits[0, -1, :]  # Last token logits
                
                # Get prediction
                probs = torch.softmax(logits_last, dim=-1)
                next_token = logits_last.argmax().item()
                confidence = probs[next_token].item()
                
                # Calculate entropy
                entropy = self.calculate_token_entropy(logits_last.unsqueeze(0))
                
                token_entropies.append(entropy)
                token_confidences.append(confidence)
            
            tgt_tokens.append(next_token)
            
            if next_token == self.sp.eos_id():
                break
        
        # Remove BOS and optionally EOS
        generated_tokens = tgt_tokens[1:-1] if tgt_tokens[-1] == self.sp.eos_id() else tgt_tokens[1:]
        translation = self.sp.decode(generated_tokens)
        
        return translation, token_entropies, token_confidences, generated_tokens
    
    def calculate_calibration_metrics(self, source_file, target_file, num_samples=100):
        """
        Calculate calibration metrics
        Calibration: do confidence scores match actual accuracy?
        """
        print(f"\nCalculating calibration metrics on {num_samples} samples...")
        
        with open(source_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f.readlines()[:num_samples]]
        
        with open(target_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f.readlines()[:num_samples]]
        
        all_confidences = []
        all_correct = []
        
        for i, (source, reference) in enumerate(zip(sources, references)):
            if i % 20 == 0:
                print(f"  Progress: {i}/{num_samples}")
            
            translation, entropies, confidences, tokens = self.translate_with_uncertainty(source)
            
            # Check token-level accuracy
            ref_tokens = self.tokenize(reference)

            # Compare up to the shorter length to avoid length-mismatch bias
            max_len = min(len(tokens), len(ref_tokens))
            for j in range(max_len):
                pred_token = tokens[j]
                conf = confidences[j]

                if pred_token == self.sp.eos_id():
                    break

                correct = (pred_token == ref_tokens[j])
                all_confidences.append(conf)
                all_correct.append(1.0 if correct else 0.0)
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(all_confidences, all_correct, num_bins=10)
        
        # Average confidence and accuracy
        avg_confidence = np.mean(all_confidences)
        avg_accuracy = np.mean(all_correct)
        
        return {
            'ece': ece,
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'total_tokens': len(all_confidences)
        }
    
    def _calculate_ece(self, confidences, correctness, num_bins=10):
        """
        Calculate Expected Calibration Error
        Groups predictions into bins and measures confidence-accuracy gap
        """
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        # Create bins
        bin_edges = np.linspace(0, 1, num_bins + 1)
        
        ece = 0.0
        for i in range(num_bins):
            # Find predictions in this bin
            in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            
            if in_bin.sum() > 0:
                bin_confidence = confidences[in_bin].mean()
                bin_accuracy = correctness[in_bin].mean()
                bin_size = in_bin.sum()
                
                # Weighted difference
                ece += (bin_size / len(confidences)) * abs(bin_confidence - bin_accuracy)
        
        return ece


def run_uncertainty_analysis(analyzer, source_file, target_file):
    """Run uncertainty quantification analysis"""
    
    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*80)
    
    # Test sentences for uncertainty analysis
    test_sentences = [
        "Ich liebe maschinelles Lernen.",
        "Das Wetter ist heute sehr schön.",
        "Die Katze schläft auf dem Sofa."
    ]
    
    print("\nPart 1: Token-level Uncertainty Analysis")
    print("-"*80)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nExample {i}: {sentence}")
        translation, entropies, confidences, tokens = analyzer.translate_with_uncertainty(sentence)
        print(f"Translation: {translation}")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print(f"Average entropy: {np.mean(entropies):.3f}")
    
    print("\n\nPart 2: Calibration Metrics")
    print("-"*80)
    
    # Calculate calibration
    calibration = analyzer.calculate_calibration_metrics(source_file, target_file, num_samples=100)
    
    print("\n" + "="*80)
    print("CALIBRATION RESULTS")
    print("="*80)
    print(f"Expected Calibration Error (ECE): {calibration['ece']:.4f}")
    print(f"Average Confidence: {calibration['avg_confidence']:.3f}")
    print(f"Average Accuracy: {calibration['avg_accuracy']:.3f}")
    print(f"Total Tokens Analyzed: {calibration['total_tokens']}")
    print()
    print("Interpretation:")
    print(f"  - ECE close to 0 = well calibrated")
    print(f"  - ECE = {calibration['ece']:.4f} -> ", end="")
    if calibration['ece'] < 0.1:
        print("Well calibrated!")
    elif calibration['ece'] < 0.2:
        print("Moderately calibrated")
    else:
        print("Poorly calibrated (overconfident or underconfident)")
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'output', 'json', 'uncertainty_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print("\n  Uncertainty analysis complete!")
    print("  - Calibration results: uncertainty_results.json")
    
    return calibration
