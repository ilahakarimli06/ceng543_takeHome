import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from question2.utils.dataset import PAD_ID
from .attentions import AdditiveAttention


class Encoder(nn.Module):
    """LSTM-based encoder for sequence-to-sequence model."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, decoder_hidden_dim, 
                 dropout, pad_id=PAD_ID, embedding_layer=None):
        super().__init__()
        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        if decoder_hidden_dim != hidden_dim:
            self.bridge_hidden = nn.Linear(hidden_dim, decoder_hidden_dim, bias=True)
            self.bridge_cell = nn.Linear(hidden_dim, decoder_hidden_dim, bias=True)
        else:
            self.bridge_hidden = None
            self.bridge_cell = None

    def forward(self, source, source_padding_mask):
        lengths = (~source_padding_mask).sum(1).cpu()
        embedded = self.embedding(source)
        embedded = self.dropout(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        encoder_outputs, (hidden, cell) = self.rnn(packed)

        # Unpack for attention mechanism - preserve original length like Question 2
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True, total_length=source.size(1))

        # Bridge encoder and decoder hidden dimensions if needed
        if self.bridge_hidden is not None:
            hidden_initial = torch.tanh(self.bridge_hidden(hidden))
            cell_initial = torch.tanh(self.bridge_cell(cell))
        else:
            hidden_initial, cell_initial = hidden, cell

        return encoder_outputs, (hidden_initial, cell_initial)


class Decoder(nn.Module):
    """LSTM-based decoder with additive attention."""
    
    def __init__(self, target_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, 
                 pad_id=PAD_ID, embedding_layer=None):
        super().__init__()
        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(encoder_dim=hidden_dim, decoder_dim=hidden_dim, 
                                          attention_dim=hidden_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim + hidden_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.output_projection = nn.Linear(hidden_dim + hidden_dim + embedding_dim, target_vocab_size)

    def forward(self, target_input, target_padding_mask, encoder_outputs, hidden_initial, source_padding_mask):
        hidden, cell = hidden_initial
        batch_size, sequence_length = target_input.shape
        outputs, attention_weights_list = [], []

        # Start with first input token (typically <sos>)
        input_token = target_input[:, 0]

        for step in range(sequence_length):
            embedded_token = self.embedding(input_token)
            embedded_token = self.dropout(embedded_token).unsqueeze(1)
            query = hidden[-1]
            context, attention_weights = self.attention(query, encoder_outputs, source_padding_mask)
            context = context.unsqueeze(1)
            rnn_input = torch.cat([embedded_token, context], dim=-1)
            rnn_output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            concatenated = torch.cat([rnn_output, context, embedded_token], dim=-1)
            logits = self.output_projection(concatenated).squeeze(1)
            outputs.append(logits)
            attention_weights_list.append(attention_weights)
            if step + 1 < sequence_length:
                input_token = target_input[:, step + 1]

        outputs = torch.stack(outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        return outputs, (hidden, cell), attention_weights


class Seq2Seq(nn.Module):

    def __init__(self, source_vocab_size, target_vocab_size,
                 embedding_dim=256, encoder_hidden_dim=512, decoder_hidden_dim=512,
                 num_layers=2, attention_type="additive", dropout=0.1,
                 source_embedding=None, target_embedding=None):

        super().__init__()
        self.encoder = Encoder(source_vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_dim=encoder_hidden_dim,
                               num_layers=num_layers,
                               decoder_hidden_dim=decoder_hidden_dim,
                               dropout=dropout,
                               embedding_layer=source_embedding)

        self.decoder = Decoder(target_vocab_size=target_vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_dim=decoder_hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               embedding_layer=target_embedding)

    def forward(self, batch):
        source = batch["src"]
        target_input = batch["target_input"]
        source_mask = batch["source_mask"]
        target_mask = batch["target_mask"]

        encoder_outputs, hidden_initial = self.encoder(source, source_mask)
        outputs, _, attention_weights = self.decoder(target_input, target_mask, encoder_outputs, 
                                                     hidden_initial, source_mask)
        return outputs, attention_weights
