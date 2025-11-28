"""
Embedding utilities for Question 3c
Aligns GloVe, BERT, and Random embeddings to SentencePiece vocabulary
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import urllib.request
import zipfile
import numpy as np
from pathlib import Path
import sentencepiece as spm


class GloVeEmbeddings:
    """Load GloVe embeddings."""
    def __init__(self, name="6B", dim=300, cache_dir=".glove_cache"):
        self.name = name
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.stoi = {}  # string to index
        self.vectors = []
        
        self._load_glove()
    
    def _load_glove(self):
        """Download and load GloVe embeddings."""
        filename = f"glove.{self.name}.{self.dim}d.txt"
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            print(f"Downloading GloVe {self.name} {self.dim}d embeddings...")
            url = f"https://nlp.stanford.edu/data/glove.{self.name}.zip"
            zip_path = self.cache_dir / f"glove.{self.name}.zip"
            
            if not zip_path.exists():
                urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(filename, self.cache_dir)
        
        print(f"Loading GloVe embeddings from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self.stoi[word] = len(self.vectors)
                self.vectors.append(vector)
        
        self.vectors = torch.tensor(np.array(self.vectors), dtype=torch.float32)
        print(f"Loaded {len(self.stoi)} GloVe vectors of dimension {self.dim}")
    
    def get_vector(self, word):
        """Get GloVe vector for a word, return None if not found."""
        word_lower = word.lower().strip()
        if word_lower in self.stoi:
            return self.vectors[self.stoi[word_lower]]
        return None


def create_glove_aligned_embedding(sp_model_path, glove_embeddings, pad_id=0):
    
    ##Create embedding matrix aligned to SentencePiece vocabulary using GloVe.

    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.vocab_size()
    emb_dim = glove_embeddings.dim
    
    print(f"Creating GloVe-aligned embeddings for {vocab_size} SentencePiece tokens...")
    
    embedding_matrix = torch.zeros(vocab_size, emb_dim)
    found_count = 0
    special_count = 0
    
    for token_id in range(vocab_size):
        piece = sp.id_to_piece(token_id)
        
        # Remove SentencePiece prefix and clean
        text = piece.replace('▁', ' ').strip()
        
        # Skip padding, empty, and special tokens
        if token_id == pad_id or not text or text in ['<s>', '</s>', '<unk>', '<pad>']:
            special_count += 1
            continue  # Keep zero vector for padding
        
        # Try direct lookup
        vec = glove_embeddings.get_vector(text)
        if vec is not None:
            embedding_matrix[token_id] = vec
            found_count += 1
        else:
            # Split into words and average
            words = text.split()
            vecs = [glove_embeddings.get_vector(w) for w in words if glove_embeddings.get_vector(w) is not None]
            if vecs:
                embedding_matrix[token_id] = torch.stack(vecs).mean(dim=0)
                found_count += 1
            else:
                # Fallback: random initialization (small values)
                embedding_matrix[token_id] = torch.randn(emb_dim) * 0.01
    
    print(f"  Matched {found_count}/{vocab_size} tokens to GloVe")
    print(f"  Special/padding tokens: {special_count}/{vocab_size}")
    
    emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
    emb.weight.data.copy_(embedding_matrix)
    return emb


def create_bert_aligned_embedding(sp_model_path, bert_model_name="bert-base-multilingual-cased", pad_id=0):
    
    #Create embedding matrix aligned to SentencePiece vocabulary using BERT.
    ##For each SP token: decode → BERT tokenize → get BERT embeddings → weighted average using attention mask.

    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.vocab_size()
    
    print(f"Loading BERT model: {bert_model_name}")
    bert_model = AutoModel.from_pretrained(bert_model_name)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model.eval()
    unk_token_id = bert_tokenizer.unk_token_id
    
    emb_dim = bert_model.config.hidden_size
    
    print(f"Creating BERT-aligned embeddings for {vocab_size} SentencePiece tokens...")
    
    embedding_matrix = torch.zeros(vocab_size, emb_dim)
    skipped_count = 0
    unk_fallback_count = 0
    
    with torch.no_grad():
        for token_id in range(vocab_size):
            piece = sp.id_to_piece(token_id)
            text = piece.replace('▁', ' ').strip()
            
            # Skip padding, empty, and special tokens
            if token_id == pad_id or not text or text in ['<s>', '</s>', '<unk>', '<pad>']:
                skipped_count += 1
                continue
            
            # Skip pure punctuation tokens (avoid noise)
            if len(text) <= 2 and all(not c.isalnum() for c in text):
                embedding_matrix[token_id] = torch.randn(emb_dim) * 0.01
                skipped_count += 1
                continue
            
            # Tokenize with BERT (no special tokens)
            encoded = bert_tokenizer(text, return_tensors='pt', add_special_tokens=False)
            input_ids = encoded['input_ids']
            
            # If BERT only produces UNK for this SentencePiece token, fall back to random to avoid collapse
            if input_ids.numel() == 0 or torch.all(input_ids == unk_token_id):
                embedding_matrix[token_id] = torch.randn(emb_dim) * 0.01
                skipped_count += 1
                unk_fallback_count += 1
                continue
            
            # Get BERT embeddings
            outputs = bert_model(**encoded)
            embeddings = outputs.last_hidden_state  # [1, seq_len, emb_dim]
            
            # Weighted average using attention mask (more robust than simple mean)
            attention_mask = encoded['attention_mask'].unsqueeze(-1).float()  # [1, seq_len, 1]
            masked_embeddings = embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)  # [1, emb_dim]
            sum_mask = attention_mask.sum(dim=1)  # [1, 1]
            embedding_matrix[token_id] = (sum_embeddings / sum_mask).squeeze(0)
    
    print(f"  BERT embeddings created (dim={emb_dim})")
    print(f"  Skipped/Special tokens: {skipped_count}/{vocab_size} (UNK fallbacks: {unk_fallback_count})")
    
    emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
    emb.weight.data.copy_(embedding_matrix)
    return emb


def create_embedding_layer(vocab_size, emb_dim, embedding_type="random", pad_id=0, 
                           sp_model_path=None, glove_embeddings=None):
    
    #Create embedding layer aligned to SentencePiece vocabulary.

    if embedding_type == "random":
        emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        print(f"Created random embeddings: {vocab_size} x {emb_dim}")
        return emb
    
    elif embedding_type == "glove":
        assert sp_model_path is not None, "SentencePiece model path required"
        assert glove_embeddings is not None, "GloVe embeddings required"
        return create_glove_aligned_embedding(sp_model_path, glove_embeddings, pad_id)
    
    elif embedding_type == "bert":
        assert sp_model_path is not None, "SentencePiece model path required"
        return create_bert_aligned_embedding(sp_model_path, pad_id=pad_id)
    
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")



