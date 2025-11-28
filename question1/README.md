# Question 1 - Comparative Analysis of Recurrent Architectures and Embedding Paradigms

## Dataset

- **IMDb**: Large Movie Review Dataset (Maas et al., 2011)

## Installation & Setup

### 1. Follow the setup instructions in the main `README.md`.

### 2. Download NLTK Data

The script automatically downloads the required NLTK tokenizers, but you can also download them manually:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## Training All Models

### Run Complete Experiment

Train all four architecture-embedding combinations:

```PowerShell
python question1\main.py
```

This will run:

1. **GloVe + BiLSTM** – Train and evaluate
2. **GloVe + BiGRU** – Train and evaluate
3. **BERT + BiLSTM** – Train and evaluate
4. **BERT + BiGRU** – Train and evaluate

### Output

All outputs are saved to `question1/output/` directory:

- Model checkpoints:  
  `output/glove__bilstm_model.pt`, `output/glove__bigru_model.pt`, `output/bert__bilstm_model.pt`, `output/bert__bigru_model.pt`
- Results:  
  `output/results.json` (contains all metrics and training history)

### Training Configuration

- **Epochs:** 10 (early stopping, patience = 3)
- **Learning Rate:**
  - Static embeddings: 1e-3
  - BERT encoder: 1e-5
  - RNN layers: 1e-3
- **Optimizer:** AdamW with weight decay (0.01)
- **Batch Size:** 32
- **Hidden Dimension:** 128
- **Max Sequence Length:** 256

---

## Results & Evaluation

After training, check `output/results.json` for detailed metrics.

---

## Visualization

### Visualize All Models

```PowerShell
python question1\visualize_embeddings.py
```

### Visualize Specific Model (Optional)

```PowerShell
python visualize_embeddings.py --model glove_bilstm
python visualize_embeddings.py --model bert_bigru
```

### Custom Parameters

```PowerShell
python visualize_embeddings.py `
    --max_samples 2000 `
    --perplexity 30 `
    --output_dir visualizations `
    --seed 42
```

### Available Models

- `glove_bilstm`: GloVe + BiLSTM
- `glove_bigru`: GloVe + BiGRU
- `bert_bilstm`: BERT + BiLSTM
- `bert_bigru`: BERT + BiGRU

### Output

Generated in `question1/visualizations/`:

- Individual plots: `{model}_pca.png`, `{model}_tsne.png`
- Combined plots: `combined_pca.png`, `combined_tsne.png`
- Metrics: `latent_visualization_metrics.json`

---

## Code Components

### `main.py`

Main orchestration script that:

- Loads data for both static and contextual modes
- Trains all 4 model combinations
- Saves checkpoints and results
- Provides progress tracking

### `model.py`

Contains 4 model classes:

```python
class BiLSTM(nn.Module):
    # GloVe embeddings + BiLSTM + FC

class BiGRU(nn.Module):
    # GloVe embeddings + BiGRU + FC

class BERTBiLSTM(nn.Module):
    # BERT encoder + BiLSTM + FC

class BERTBiGRU(nn.Module):
    # BERT encoder + BiGRU + FC
```

All models include `get_features()` method for extracting hidden representations.

### `train.py`

Training loop with:

- Gradient clipping (max_norm=1.0)
- Early stopping (patience=3)
- Metric tracking (accuracy, macro F1)
- Epoch timing for convergence analysis

### `data_loader.py`

Unified data loading for both modes:

- **Static mode**: Downloads GloVe, tokenizes with NLTK, creates embedding matrix
- **Contextual mode**: Uses BERT tokenizer, dynamic batching
- Handles IMDb dataset loading from HuggingFace
- Supports train/validation/test splits

### `utils.py`

Utility functions:

- `set_seed()`: Reproducibility
- `EpochTimer`: Measure training time
- `prepare_batch()`: Mode-agnostic batch preprocessing
- `project_2d()`: PCA/t-SNE projection
- `analyze_separability()`: Clustering metrics
- Plotting utilities

### `visualize_embeddings.py`

Visualization pipeline:

- Loads trained checkpoints
- Extracts hidden representations
- Applies PCA and t-SNE
- Generates publication-quality plots
- Computes separability metrics

---

## CUDA Errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
# Edit main.py: device = torch.device("cpu")
```

### GloVe Download Issues

If GloVe download fails:

1. Manually download from: https://nlp.stanford.edu/data/glove.6B.zip
2. Extract `glove.6B.300d.txt` to `.glove_cache/`

## Citation & References

### Datasets

- **IMDb**: Maas et al. (2011) - Large Movie Review Dataset
- **GloVe**: Pennington et al. (2014) - GloVe: Global Vectors for Word Representation
- **BERT**: Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers

### Models

- **LSTM**: Hochreiter & Schmidhuber (1997)
- **GRU**: Cho et al. (2014)
- **Bidirectional RNNs**: Schuster & Paliwal (1997)

---

## 📧 Notes

- All experiments are reproducible with `seed=42`
- Training time varies based on hardware (GPU recommended)
- Results are saved automatically after each model
- Checkpoints include full model state for later analysis
