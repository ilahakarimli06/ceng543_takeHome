## Question 3 – Transition from Recurrent to Transformer Architectures

All models use the **identical dataset** from Question 2 (Multi30k English-German translation)
with the same tokenization (SentencePiece) and comparable hyperparameters.

### Quick Start

**Note**: Make sure you have completed Question 2 data preparation (dataset tokenization with SentencePiece).

#### 1. Train Seq2Seq with Different Embeddings

```bash
python question3/train.py
```

This trains three Seq2Seq models (random, GloVe, BERT embeddings) and reports:

- Perplexity on validation set
- BLEU and ROUGE scores on test set
- Training time and GPU memory usage

#### 2. Train Transformer with Different Embeddings

```bash
python question3/train_transformer.py
```

#### 3. Run Ablation Study

```bash
python question3/ablation_study.py
```

Tests different configurations:

- Layers: 1, 2, 4 layers
- Attention Heads: 2, 4, 8 heads

### Evaluation Metrics

All models are evaluated on:

- Perplexity: Lower is better (measures prediction confidence)
- BLEU: Translation quality (corpus-level metric)
- ROUGE-1/2/L: N-gram overlap with reference translations
- Training Time: Seconds per epoch
- GPU Memory: Peak memory usage during training

### Embedding Configurations

All embeddings are **aligned to the same SentencePiece vocabulary** from Question 2:

**Random Embeddings (256d)**

- Standard `nn.Embedding` with Random(default for python) initialization
- Learned from scratch during training

**GloVe Embeddings (300d)** - This decision was made because Question 3 requires the use of a comparable tokenizer.

- Each SentencePiece token decoded to text
- Lookup in GloVe vocabulary
- Subwords averaged if no direct match
- Pre-initialized but trainable

**BERT Embeddings (768d)**

- Each SentencePiece token decoded to text
- Re-tokenized with BERT tokenizer
- BERT embeddings extracted and averaged
- Pre-initialized but trainable

SentencePiece cannot be used directly, so this will make tokenizers comparable

---
