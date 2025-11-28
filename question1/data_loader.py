import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Tuple, Optional, Dict, Any
from nltk.tokenize import  word_tokenize
import nltk
# Download required NLTK data files
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except:
    pass  # Already downloaded or download failed

# For static embedding(GloVe) - manual loading because of torchtext compatibility issues
## Suggested by ChatGPT
import os
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

# For contectual embedding(BERT)
from transformers import AutoTokenizer, DataCollatorWithPadding

# Import set_seed from utils (more complete implementation)
from utils import set_seed


# ---------------------------------------------------------
# GloVe Embeddings Loader (without torchtext)
## Help of gpt üas used for syntax
# ---------------------------------------------------------
class GloVeEmbeddings:
    """Manually load GloVe embeddings without torchtext dependency."""
    def __init__(self, name="6B", dim=300, cache_dir=".glove_cache"):
        self.name = name
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.stoi = {}  # string to index
        self.vectors = []  # list of vectors
        
        self._load_glove()
    
    def _load_glove(self):
        """Download and load GloVe embeddings."""
        filename = f"glove.{self.name}.{self.dim}d.txt"
        filepath = self.cache_dir / filename
        
        # Download if not exists
        if not filepath.exists():
            print(f"Downloading GloVe {self.name} {self.dim}d embeddings...")
            # GloVe 6B zip contains all dimensions (50d, 100d, 200d, 300d)
            url = f"https://nlp.stanford.edu/data/glove.{self.name}.zip"
            zip_path = self.cache_dir / f"glove.{self.name}.zip"
            
            # Download the zip file
            if not zip_path.exists():
                print(f"Downloading from {url}...")
                urllib.request.urlretrieve(url, zip_path)
                print("Download complete!")
            
            # Extract only the needed file from zip
            print(f"Extracting {filename} from zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(filename, self.cache_dir)
            print("Extraction complete!")
        
        # Load embeddings
        print(f"Loading GloVe embeddings from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self.stoi[word] = idx
                self.vectors.append(vector)
        
        self.vectors = torch.tensor(np.array(self.vectors), dtype=torch.float32)
        print(f"Loaded {len(self.stoi)} GloVe vectors of dimension {self.dim}")


# ---------------------------------------------------------
# Dataset
# mode = "static"  - GloVe
# mode = "contextual" - BERT tokenizer IDs
# ---------------------------------------------------------
class TextDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        mode: str = "static",
        max_len: int = 256, #changed from 512 to 256 to reduce memory usage
        vocab: Optional[GloVeEmbeddings] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        lowercase: bool = True,
    ):
        assert mode in ("static", "contextual"), "mode must be 'static' or 'contextual'"
        self.texts = texts
        self.labels = labels
        self.mode = mode
        self.max_len = max_len
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.lowercase = lowercase

        if self.mode == "static":
            assert self.vocab is not None, "Static mode requires a GloVe vocab"
            
            self.emb_dim = self.vocab.dim
        else:
            assert self.tokenizer is not None, "Contextual mode requires a HF tokenizer"

    def _tokenize_text(self, text: str):
        # Very simple tokenizer using NLTK
        return word_tokenize(text)

    def _vectorize_static(self, text: str) -> Tuple[torch.Tensor, int]:
        
        # Convert a text string to a sequence of GloVe vectors (static embedding).
        # Pads or truncates to max_len. Returns tensor and real length.

        # Lowercase the text - GloVe 6B is case-sensitive
        if self.lowercase:
            text = text.lower()

        tokens = self._tokenize_text(text)
        vecs = []

        # Take first max_len tokens, put GloVe vector; if not, put zero vector
        for t in tokens[: self.max_len]:
            idx = self.vocab.stoi.get(t, None)  # index of the token in the vocabulary
            if idx is not None:
                vecs.append(self.vocab.vectors[idx].clone())  # vector of the token
            else:
                vecs.append(torch.zeros(self.vocab.dim))

        # Save the actual number of tokens
        real_length = len(vecs)
        
        # If short, fill with PAD (zero vector)
        if len(vecs) < self.max_len:
            pad_needed = self.max_len - len(vecs)
            vecs += [torch.zeros(self.vocab.dim)] * pad_needed

        return torch.stack(vecs), real_length  # stack usage looked from the internet

    def _tokenize_contextual(self, text: str) -> Dict[str, torch.Tensor]:
    
        # Tokenize text using HuggingFace tokenizer for contextual embeddings (BERT).
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        # return_tensors="pt" -> shape [1, seq_len]; squeeze to [seq_len] Suggested by ChatGPT
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        # For BERT: if token_type_ids is not in enc, fill with zeros
        if "token_type_ids" not in enc:
            enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

        return enc

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx]).long() #loss function requires long type

        if self.mode == "static":
            x, length = self._vectorize_static(text)  # [max_len, emb_dim], int
            return {"inputs": x, "lengths": torch.tensor(length)}, label

        else:  # contextual
            enc = self._tokenize_contextual(text) 
            return enc, label

    def __len__(self) -> int:
        return len(self.texts)


# ---------------------------------------------------------
# Helper: prepare IMDb dataset (train/val/test)
# ---------------------------------------------------------
def _prepare_imdb_splits(limit_train: int = 5000, limit_test: int = 1000, val_ratio: float = 0.1):
    ds = load_dataset("imdb")

    # IMPORTANT: Shuffle dataset before sampling to get balanced labels
    # After first try I noticed that the dataset is not balanced, so I shuffled it to get balanced labels
    # I know that IMDB dataset is ordered: first 12.5k are negative, last 12.5k are positive, but I shuffled it to get balanced labels
    train_ds = ds["train"].shuffle(seed=42)
    train_texts = train_ds["text"][:limit_train]
    train_labels = train_ds["label"][:limit_train]

    test_ds = ds["test"].shuffle(seed=42)
    test_texts = test_ds["text"][:limit_test]
    test_labels = test_ds["label"][:limit_test]

    # Validation split (last %10 of train)
    n_train = len(train_texts)
    n_val = int(n_train * val_ratio)
    val_texts = train_texts[-n_val:]
    val_labels = train_labels[-n_val:]
    train_texts = train_texts[:-n_val]
    train_labels = train_labels[:-n_val]

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def get_dataloaders(
    mode: str = "static",
    batch_size: int = 32,
    max_len: int = 256,
    seed: int = 42,
    limit_train: int = 5000,
    limit_test: int = 1000,
    model_name: str = "bert-base-uncased",
    num_workers: int = 2,
    pin_memory: bool = True,
):
    set_seed(seed)

    # IMDb split
    (tr_texts, tr_labels), (va_texts, va_labels), (te_texts, te_labels) = _prepare_imdb_splits(
        limit_train=limit_train, limit_test=limit_test
    )

    if mode == "static":
        vocab = GloVeEmbeddings(name="6B", dim=300)  # Because of the assignment requirements
        tokenizer = None
    else:
        vocab = None
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = TextDataset(
        tr_texts, tr_labels, mode=mode, max_len=max_len, vocab=vocab, tokenizer=tokenizer
    )
    val_ds = TextDataset(
        va_texts, va_labels, mode=mode, max_len=max_len, vocab=vocab, tokenizer=tokenizer
    )
    test_ds = TextDataset(
        te_texts, te_labels, mode=mode, max_len=max_len, vocab=vocab, tokenizer=tokenizer
    )


    if mode == "contextual":

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    else:

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    meta = {
        "mode": mode,
        "max_len": max_len,
        "embedding_dim": (300 if mode == "static" else None),
        "model_name": (None if mode == "static" else model_name),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    return train_loader, val_loader, test_loader, meta


