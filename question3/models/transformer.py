import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model based on 'Attention Is All You Need' (Vaswani et al., 2017)."""
    
    def __init__(self, source_vocab_size, target_vocab_size, model_dim=256, num_heads=8, 
                 num_encoder_layers=2, num_decoder_layers=2, feedforward_dim=512, 
                 dropout=0.1, pad_id=0, source_embedding=None, target_embedding=None):
        super().__init__()
        
        self.model_dim = model_dim
        self.pad_id = pad_id
        self._attn_maps = []
        self._attn_hooks = []
        
        # Embeddings
        if source_embedding is not None:
            self.source_embedding = source_embedding
        else:
            self.source_embedding = nn.Embedding(source_vocab_size, model_dim, padding_idx=pad_id)
        if target_embedding is not None:
            self.target_embedding = target_embedding
        else:
            self.target_embedding = nn.Embedding(target_vocab_size, model_dim, padding_idx=pad_id)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim, dropout=dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(model_dim, target_vocab_size)
        
        self._init_weights()
        self._register_attention_hooks()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, size):
        """Create a causal (subsequent) mask for the target of shape (size, size).

        Returns a float mask where positions that should be masked (future
        tokens) contain -inf and others contain 0.0. This is the numeric form
        expected by `nn.Transformer` for `tgt_mask` so that softmax sets
        those entries to zero probability.
        """

        mask = torch.triu(torch.ones(size, size), diagonal=1)
        # Replace 1.0 with -inf (masked) and 0.0 stays 0.0 (unmasked)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, 0.0)
        return mask
    
    def forward(self, batch):
        source = batch["src"]
        target_input = batch["target_input"]
        source_mask = batch["source_mask"]
        target_mask = batch["target_mask"]

        # Clear previous stored attention maps
        self._attn_maps = []

        # Create causal mask for target
        target_length = target_input.size(1)
        target_subsequent_mask = self.generate_square_subsequent_mask(target_length).to(target_input.device)

        # Embeddings with scaling (as per Vaswani et al., 2017)
        source_embedded = self.source_embedding(source) * math.sqrt(self.model_dim)
        target_embedded = self.target_embedding(target_input) * math.sqrt(self.model_dim)

        # Add positional encoding
        source_embedded = self.positional_encoding(source_embedded)
        target_embedded = self.positional_encoding(target_embedded)

        # Transformer forward pass
        output = self.transformer(
            source_embedded,
            target_embedded,
            tgt_mask=target_subsequent_mask,
            src_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=source_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)

        # Return logits and collected encoder-decoder attentions
        # self._attn_maps is a list of tensors (one per decoder layer)
        return logits, list(self._attn_maps)

    # ------------------------------
    # Attention capture utilities
    # ------------------------------
    def _register_attention_hooks(self):
        """Register forward hooks to save decoder cross-attention weights."""
        if not hasattr(self.transformer, "decoder"):
            return
        for layer in self.transformer.decoder.layers:
            hook = layer.multihead_attn.register_forward_hook(self._save_cross_attention)
            self._attn_hooks.append(hook)

    def _save_cross_attention(self, module, inputs, output):
        """
        Forward hook to stash encoder-decoder attention.
        output is (attn_output, attn_weights)
        """
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None:
                # Detach to avoid holding graph; keep on CPU for visualization
                self._attn_maps.append(attn_weights.detach().cpu())
