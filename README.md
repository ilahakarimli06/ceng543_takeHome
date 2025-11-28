# CENG543 Take‑Home – Root Guide

This repository contains five question folders (`question1` … `question5`) with their own READMEs and commands. This file only gives a top-level setup/run guide without altering per-question instructions.

## 1) Environment

- Requirements: Python 3.10+, `git` (CUDA optional; PyTorch will fall back to CPU if needed).
- Package manager: `uv` recommended (works with requirements.txt and uv.lock).
  ```bash
  pip install uv
  ```

## 2) Install Dependencies

- From repo root:
  ```bash
  # create and activate a virtualenv
  uv venv
  .\.venv\Scripts\activate      # Windows PowerShell
  # or: source .venv/bin/activate (Unix)
  uv sync                       # installs from uv.lock / requirements.txt
  ```
- If you prefer pip:
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install -r requirements.txt
  ```

## 3) Data & Models

- SentencePiece model and token id files live under `question2/data/`.
- Pretrained checkpoints are already present (e.g., `question2/output/*.pt`, `question5/best_model_4l_2h.pt`), so you can run analyses without retraining.

## 4) Running

- Each question folder has its own README/commands; follow those for details.
- Quick example (Question 5 end-to-end analyses):
  ```bash
  cd question5
  python main.py
  ```
- For training/visualization variants, see the specific question READMEs.

## 5) Troubleshooting

- Ensure the venv is active (`which python` or `Get-Command python` should point to `.venv`).
- CUDA warnings are benign; code will run on CPU if GPU is unavailable.
- Missing package? From repo root, rerun `uv sync` (or `pip install -r requirements.txt`).

This is a general root guide; use per-question READMEs for experiment-specific commands and details.

## AI Usage Declaration

This project was completed with the assistance of AI tools in the following capacities:

- **Code debugging and optimization**: GitHub Copilot was used to debug PyTorch errors and optimize training loops
- **Documentation and report writing**: AI assisted in structuring LaTeX sections and improving academic English phrasing

All core implementations (model architectures, training pipelines, experimental design) and interpretations of results are original work.
