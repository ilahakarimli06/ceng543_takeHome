"""
Question 5 - Model Analyzer (Shared utilities)
Common model loading and translation utilities for interpretability and failure analysis
"""

import torch
import sys
import os
import numpy as np
import warnings
import sentencepiece as spm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from question3.evaluate import greedy_decode_transformer_with_attention
from question3.evaluate import greedy_decode_transformer
# Use fixed transformer that properly captures attention weights
from question5.models.transformer_fixed import TransformerModel


class ModelAnalyzer:
    """Base class for model analysis with shared utilities"""

    def __init__(self, model_path, vocab_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_path)

        # Load model
        self.model, self.config = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        """Load trained transformer model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get model config from checkpoint
        config = {
            'vocab_size': checkpoint.get('vocab_size', 8000),
            'model_dim': checkpoint.get('d_model', 256),
            'num_heads': checkpoint.get('nhead', 2),
            'num_layers': checkpoint.get('num_layers', 4),
            'feedforward_dim': checkpoint.get('feedforward_dim', 512),  # Ablation study uses 512
            'embedding_type': checkpoint.get('embedding_type', 'random')
        }

        # Create model with correct parameter names
        model = TransformerModel(
            source_vocab_size=config['vocab_size'],
            target_vocab_size=config['vocab_size'],
            model_dim=config['model_dim'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'],
            feedforward_dim=config['feedforward_dim'],
            dropout=0.1,
            pad_id=0
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model, config

    def tokenize(self, text):
        """Tokenize text using sentencepiece"""
        return self.sp.encode(text, out_type=int)

    def detokenize(self, ids):
        """Convert token ids back to text"""
        return self.sp.decode(ids)

    def translate(self, source_text, max_len=50):
        """Translate German to English using Question 3's decoder"""

        src_tokens = self.tokenize(source_text)
        predicted_ids = greedy_decode_transformer(
            self.model, src_tokens, self.sp, max_len, self.device
        )
        translation = self.sp.decode(predicted_ids)
        return translation

    def translate_and_get_attention(self, source_text, max_len=50):
        """
        Translate text and extract attention weights
        Returns: translation, attention_weights, src_tokens, tgt_tokens
        """
    
        # Tokenize and translate
        src_tokens = self.tokenize(source_text)
        tgt_tokens, attention_weights = greedy_decode_transformer_with_attention(
            self.model, src_tokens, self.sp, max_len, self.device
        )
        translation = self.sp.decode(tgt_tokens)
    
        # Convert attention to numpy array (tgt_len x src_len)
        if attention_weights is not None:
            if torch.is_tensor(attention_weights):
                attention_weights = attention_weights.detach().cpu().numpy()
        else:
            # Should not happen with properly patched model
            warnings.warn("Attention weights are None - model may not be collecting attention properly")
            attention_weights = np.ones((len(tgt_tokens), len(src_tokens))) / max(1, len(src_tokens))
    
        return translation, attention_weights, src_tokens, tgt_tokens

