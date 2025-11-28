import torch
import torch.nn as nn


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax with masking for padding positions.

    """
    mask = mask.to(torch.bool)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


class AdditiveAttention(nn.Module):
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, encoder_outputs: torch.Tensor, source_padding_mask: torch.Tensor):
        """
            query: Decoder hidden state (batch_size, decoder_dim)
            encoder_outputs: Encoder outputs (batch_size, source_length, encoder_dim)
            source_padding_mask: Padding mask (batch_size, source_length)
        """
        encoder_proj = self.encoder_projection(encoder_outputs)
        decoder_proj = self.decoder_projection(query).unsqueeze(1)
        scores = self.attention_vector(torch.tanh(encoder_proj + decoder_proj)).squeeze(-1)
        attention_weights = masked_softmax(scores, source_padding_mask)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights




