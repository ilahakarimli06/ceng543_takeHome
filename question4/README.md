# Question 4 - Retrieval-Augmented Generation (RAG)

**Environment / HuggingFace token** - IMPORTANT

- You need a Hugging Face user access token to download some datasets (used in `load_hotpotqa.py`). Set it via an environment variable named `HF_TOKEN`.

**Temporary / session usage**

- Instead of `.env`, you can set the token for a single PowerShell session:

```powershell
$env:HF_TOKEN = "hf_your_real_token_here"
python question4\evaluate.py
```

## Components

### Retrievers

- **TF-IDF** (`tf_idf.py`) - Sparse retriever using term frequency
- **Sentence-BERT** (`sbert.py`) - Dense retriever using embeddings

### Generator

- **FLAN-T5** (`flan_t5_rag.py`) - Neural text generation model

### Evaluation

- **Retrieval Metrics**: Precision@k, Recall@k
- **Generation Metrics**: BLEU, ROUGE-L, BERTScore
- **Components**: Retrieval-only, Generation-only (gold context), Joint RAG

## Usage

### From PowerShell

```PowerShell
python question4\evaluate.py
```

### From Python

```python
from evaluate import run_evaluation

# Run full evaluation with 50 test samples
run_evaluation(num_test=50)
```

## Requirements

```
datasets
transformers
torch
sentence-transformers
scikit-learn
nltk
rouge-score
bert-score
```
