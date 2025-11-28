## To Start

1. Follow the instructions on main README.md

## Used DataSet

- **Multi30k**: Multilingual Image Description Dataset (Elliott et al., 2016)

## Data Cleaning

First run: uv run python question2/utils/data_loader2.py (downloads train/validation/test into question2/data/raw)

Then normalize the files:

```PowerShell

uv run python question2/utils/preprocess.py `
  --input_dir question2/data/raw `
  --out_dir question2/data/cleaned `
  --chunksize 20000
```

## Train shared SentencePiece tokenizer

```PowerShell
uv run python question2/utils/sentence_piece.py `
  --input_glob "question2/data/cleaned/train.*" `
  --model_prefix question2/data/spm_shared_unigram `
  --vocab_size 8000
```

## Convert text to token IDs

```PowerShell
python question2/utils/data_translation.py --spm question2/data/spm_shared_unigram.model --in question2/data/cleaned/train.en --out question2/data/tokenized/train.en.ids

python question2/utils/data_translation.py --spm question2/data/spm_shared_unigram.model --in question2/data/cleaned/validation.en --out question2/data/tokenized/validation.en.ids

python question2/utils/data_translation.py --spm question2/data/spm_shared_unigram.model --in question2/data/cleaned/test.en --out question2/data/tokenized/test.en.ids

python question2/utils/data_translation.py --spm question2/data/spm_shared_unigram.model --in question2/data/cleaned/train.de --out question2/data/tokenized/train.de.ids

python question2/utils/data_translation.py --spm question2/data/spm_shared_unigram.model --in question2/data/cleaned/validation.de --out question2/data/tokenized/validation.de.ids

python question2/utils/data_translation.py --spm question2/data/spm_shared_unigram.model --in question2/data/cleaned/test.de --out question2/data/tokenized/test.de.ids
```

## Train, Evaluate & Compare Attention Mechanisms

```PowerShell
python question2/main.py
```

- Save results to `question2/output/attention_comparison_results.json`
- Save models to `question2/output/{additive,multiplicative,scaled_dot}_model.pt`
- Save attention maps to `question2/output/attn_maps/`

## Visualize Attention (Optional)

To visualize attention for a specific model:

```PowerShell
python question2/viz.py question2/output/additive_model.pt question2/output/attn_maps/additive
```
