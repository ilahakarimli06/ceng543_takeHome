from interpretability_analysis import run_interpretability_analysis
from failure_analysis import run_failure_analysis
from uncertainty_analysis import run_uncertainty_analysis, UncertaintyAnalyzer
from model_analyzer import ModelAnalyzer
from interpretability_analysis import InterpretabilityAnalyzer
from failure_analysis import FailureAnalyzer
import os, json
def main():

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    model_path = os.path.join(script_dir, 'best_model_4l_2h.pt')
    vocab_path = os.path.join(base_dir, 'question2/data/spm_shared_unigram.model')
    source_file = os.path.join(base_dir, 'question2/data/cleaned/validation.de')
    target_file = os.path.join(base_dir, 'question2/data/cleaned/validation.en')
    ablation_path = os.path.join(base_dir, 'question3/output/ablation_results.json')
    
    print("\n" + "="*80)
    print("QUESTION 5 - MODEL ANALYSIS")
    print("="*80)

    print("\n" + "="*80)
    
    # Load model once (shared between analyses)
    print("\n[0/2] Loading model...")
    print("-"*80)

    
    # Part 1: Interpretability Analysis
    print("\n[1/3] Running Interpretability Analysis...")
    print("-"*80)
    interp_analyzer = InterpretabilityAnalyzer(model_path, vocab_path)
    run_interpretability_analysis(interp_analyzer)
    
    # Part 2: Failure Case Analysis
    print("\n[2/3] Running Failure Case Analysis...")
    print("-"*80)
    failure_analyzer = FailureAnalyzer(model_path, vocab_path)
    failure_cases = run_failure_analysis(
        failure_analyzer, source_file, target_file
    )
    
    # Part 3: Uncertainty Quantification
    print("\n[3/3] Running Uncertainty Quantification...")
    print("-"*80)
    uncertainty_analyzer = UncertaintyAnalyzer(model_path, vocab_path)
    uncertainty_results = run_uncertainty_analysis(
        uncertainty_analyzer, source_file, target_file
    )
    
    # Summary
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - output/visualizations/attention_map_*.png     (5 attention heatmaps)")
    print("  - output/visualizations/ig_importance_*.png     (3 integrated gradients)")
    print("  - output/json/failure_analysis_results.json     (detailed failure analysis)")
    print("  - output/json/uncertainty_results.json          (calibration metrics)")
    print("\nModel Configuration:")
    config = interp_analyzer.config
    print(f"  - Layers: {config['num_layers']}")
    print(f"  - Attention Heads: {config['num_heads']}")
    print(f"  - Embedding: {config['embedding_type'].title()}")
    print(f"  - Model Dimension: {config['model_dim']}")
    print(f"  - Feedforward Dimension: {config['feedforward_dim']}")

    # Report best ablation score (question3)
    try:
        with open(ablation_path, 'r') as f:
            ablations = json.load(f)
        best_cfg, best_bleu = max(
            ablations.items(), key=lambda kv: kv[1].get('bleu', 0)
        )
        best_ppl = ablations[best_cfg].get('perplexity', '?')
        print(f"\nBest config from Question3 ablation: {best_cfg} | BLEU={best_bleu:.2f} | PPL={best_ppl}")
        print(f"Loaded checkpoint: {os.path.basename(model_path)} (matching 4 layers / 2 heads)")
    except Exception:
        pass
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
