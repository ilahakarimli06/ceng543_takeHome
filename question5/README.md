# Question 5 - Interpretability Analysis

## Model Selection

**Best Model**: Transformer + Random Embedding (4 layers, 2 heads)

- BLEU: 35.92 (highest from ablation study)
- Perplexity: 5.36 (lowest from ablation study)
- Configuration: 4 encoder layers, 4 decoder layers, 2 attention heads, d_model=256
- Task: German to English Translation

## Files

- `train_best_model.py` - Training script for best configuration
- `model_analyzer.py` - **Shared base class** for model loading and translation
- `interpretability_analysis.py` - Attention visualization (inherits from ModelAnalyzer)
- `failure_analysis.py` - Failure case analysis (inherits from ModelAnalyzer)
- `main.py` - **Main script** that runs all analyses
- `best_model_4l_2h.pt` - Trained model checkpoint (created after training)
- `visualizations/` - Output folder for attention heatmaps,ig_importance
- `failure_analysis_results.json` - Detailed failure case analysis results

## Usage

```bash
cd question5

# Step 1: Train the best model configuration (4 layers, 2 heads)
python train_best_model.py

# Step 2: Run ALL analyses
python main.py


```
