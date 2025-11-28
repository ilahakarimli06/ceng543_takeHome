# Question 4 - Retrieval-Augmented Generation (RAG)

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
