"""
Question 5 - Failure Case Analysis
Identify and analyze 5 representative failure cases
"""
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from model_analyzer import ModelAnalyzer
import json
import os
import numpy as np
import random

class FailureAnalyzer(ModelAnalyzer):
    """Analyze failure cases in translation"""
    
    def check_attention_misalignment(self, source, hypothesis):
        """
        Check if attention weights show misalignment
        Returns True if attention seems misaligned
        """
        try:
            # Get attention weights
            translation, attention, src_tokens, tgt_tokens = self.translate_and_get_attention(source)
            
            # Check if attention is too uniform (not focused)
            # Good attention should have peaks, not be flat
            
            if attention.shape[0] == 0 or attention.shape[1] == 0:
                return False
            
            # Calculate attention entropy (high entropy = uniform = bad alignment)
            # For each target token, check if attention is too spread out
            max_attentions = attention.max(axis=1)  # Max attention for each target token
            avg_max_attention = max_attentions.mean()
            
            # If average max attention < 0.3, attention is too diffuse
            if avg_max_attention < 0.3:
                return True  # Misaligned
            
            # Check if attention is monotonic (should roughly follow diagonal)
            # Simple check: correlation between position
            if len(src_tokens) > 3 and len(tgt_tokens) > 3:
                # Expected: target position i should attend most to source position ~i
                expected_positions = []
                actual_positions = []
                
                for tgt_idx in range(min(len(tgt_tokens), attention.shape[0])):
                    expected_pos = (tgt_idx / len(tgt_tokens)) * len(src_tokens)
                    actual_pos = attention[tgt_idx].argmax()
                    expected_positions.append(expected_pos)
                    actual_positions.append(actual_pos)
                
                # Calculate how far off the attention is
                position_diff = np.abs(np.array(expected_positions) - np.array(actual_positions))
                avg_diff = position_diff.mean()
                
                # If average difference > 50% of source length, misaligned
                if avg_diff > len(src_tokens) * 0.5:
                    return True
            
            return False
            
        except Exception as e:
            # If attention extraction fails, skip this check
            return False
    
    def calculate_bleu(self, reference, hypothesis):
        """Calculate sentence-level BLEU score"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return score * 100
    
    def analyze_failure_type(self, source, reference, hypothesis):
        """
        Classify failure type with simple heuristics
        
        Thresholds are chosen based on common translation error patterns
        """
        src_tokens = source.lower().split()
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        failure_types = []
        
        # 1. Length mismatch - translation too short or too long
        LENGTH_SHORT_RATIO = 0.5  # Less than 50% of reference length
        LENGTH_LONG_RATIO = 1.5   # More than 150% of reference length
        
        if len(hyp_tokens) < len(ref_tokens) * LENGTH_SHORT_RATIO:
            failure_types.append("Too Short Translation")
        elif len(hyp_tokens) > len(ref_tokens) * LENGTH_LONG_RATIO:
            failure_types.append("Too Long Translation")
        
        # 2. Rare words - uncommon tokens in vocabulary
        # SentencePiece vocab: lower IDs = more frequent
        RARE_TOKEN_THRESHOLD = 1000  # Tokens with ID > 1000 are less common
        
        src_ids = self.sp.encode(source, out_type=int)
        rare_count = sum(1 for tid in src_ids if tid > RARE_TOKEN_THRESHOLD)
        if rare_count > 0:
            failure_types.append(f"Rare Words ({rare_count} uncommon tokens)")
        
        # 3. Missing key words - reference words not in prediction
        MISSING_WORDS_RATIO = 0.3  # More than 30% of reference words missing
        
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)
        missing_words = ref_set - hyp_set
        if len(missing_words) > len(ref_tokens) * MISSING_WORDS_RATIO:
            failure_types.append(f"Missing Key Words ({len(missing_words)} words)")
        
        # 4. Repetition - same words repeated multiple times
        MIN_LENGTH_FOR_REPETITION = 5  # Only check if output has at least 5 words
        REPETITION_THRESHOLD = 0.7     # Less than 70% unique words = repetitive
        
        if len(hyp_tokens) > MIN_LENGTH_FOR_REPETITION:
            unique_ratio = len(set(hyp_tokens)) / len(hyp_tokens)
            if unique_ratio < REPETITION_THRESHOLD:
                failure_types.append("Repetitive Output")
        
        # 5. Long/complex source - ambiguous context
        LONG_SOURCE_THRESHOLD = 20  # More than 20 words = complex
        
        if len(src_tokens) > LONG_SOURCE_THRESHOLD:
            failure_types.append("Long/Complex Source")
        
        # 6. Attention misalignment - check actual attention weights
        # Use model's attention to see if it's properly aligned to source
        if self.check_attention_misalignment(source, hypothesis):
            failure_types.append("Attention Misalignment (diffuse or non-monotonic attention)")
        
        if not failure_types:
            failure_types.append("General Translation Error")
        
        return failure_types
    
    def find_failure_cases(self, source_file, target_file, num_cases=100):
        """Find worst translation cases"""
        print("\nFinding failure cases...")
        
        with open(source_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f.readlines()]
        
        with open(target_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f.readlines()]

        # Random sample to avoid first-N bias (deterministic seed for reproducibility)
        random.seed(42)
        indices = random.sample(range(len(sources)), k=min(num_cases, len(sources)))
        print(f"Analyzing {len(indices)} samples (random subset)...")
        
        failures = []
        
        for idx, i in enumerate(indices):
            if idx % 20 == 0:
                print(f"  Progress: {idx}/{len(indices)}")
            
            source = sources[i]
            reference = references[i]
            
            # Translate
            hypothesis = self.translate(source)
            
            # Calculate BLEU
            bleu = self.calculate_bleu(reference, hypothesis)
            
            failures.append({
                'index': i,
                'source': source,
                'reference': reference,
                'hypothesis': hypothesis,
                'bleu': bleu
            })
        
        # Sort by BLEU (worst first)
        failures.sort(key=lambda x: x['bleu'])
        
        return failures
    
    def analyze_top_failures(self, failures, top_n=5):
        """Analyze top N failure cases"""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} FAILURE CASES ANALYSIS")
        print(f"{'='*80}\n")
        
        analyzed_cases = []
        
        for i, case in enumerate(failures[:top_n], 1):
            print(f"{'='*80}")
            print(f"FAILURE CASE #{i}")
            print(f"{'='*80}")
            print(f"Source (DE):     {case['source']}")
            print(f"Reference (EN):  {case['reference']}")
            print(f"Predicted (EN):  {case['hypothesis']}")
            print(f"BLEU Score:      {case['bleu']:.2f}")
            print()
            
            # Analyze failure type
            failure_types = self.analyze_failure_type(
                case['source'], 
                case['reference'], 
                case['hypothesis']
            )
            
            print("Failure Analysis:")
            for ftype in failure_types:
                print(f"  - {ftype}")
            
            # Additional diagnostics
            print(f"\nDiagnostics:")
            print(f"  Source length:     {len(case['source'].split())} words")
            print(f"  Reference length:  {len(case['reference'].split())} words")
            print(f"  Prediction length: {len(case['hypothesis'].split())} words")
            
            src_tokens = self.sp.encode(case['source'], out_type=int)
            print(f"  Source tokens:     {len(src_tokens)} tokens")
            print(f"  Vocab coverage:    {sum(1 for t in src_tokens if t < 1000)}/{len(src_tokens)} common tokens")
            
            print()
            
            analyzed_cases.append({
                'case_number': i,
                'source': case['source'],
                'reference': case['reference'],
                'hypothesis': case['hypothesis'],
                'bleu': case['bleu'],
                'failure_types': failure_types,
                'source_length': len(case['source'].split()),
                'reference_length': len(case['reference'].split()),
                'prediction_length': len(case['hypothesis'].split())
            })
        
        return analyzed_cases


def run_failure_analysis(analyzer, source_file, target_file):
    """Run failure case analysis with existing analyzer"""
    
    print("\n" + "="*80)
    print("FAILURE CASE ANALYSIS")
    print("="*80)
    
    # Find failure cases
    failures = analyzer.find_failure_cases(source_file, target_file, num_cases=100)
    
    # Analyze top 5 failures
    analyzed = analyzer.analyze_top_failures(failures, top_n=5)
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'output', 'json', 'failure_analysis_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(analyzed, f, indent=2, ensure_ascii=False)
    print("\n" + "="*80)
    print("Failure analysis complete!")
    print(f" Results saved to: {out_path}")
    print("="*80)
    
    return analyzer, analyzed
