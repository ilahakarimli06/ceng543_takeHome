import torch
import torch.nn as nn
from utils.dataset import PAD_ID
from .attentions import AdditiveAttention, MultiplicativeAttention, ScaledDotProductAttention

class Encoder(nn.Module):
    """
    forward(src, source_mask) -> (enc_outs, dec_init)
      src: (Batch_size, Sequence_length) token ids
      source_mask: (Batch_size, Sequence_length) bool, True=PAD
      enc_outs: (Batch_size, Sequence_length, hid_dim)
      h0, c0: (num_layers, Batch_size, dec_hid) initial decoder states
    """

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dec_hid, dropout, pad_id=PAD_ID):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)

        dec_hid = dec_hid or hid_dim
        if dec_hid != hid_dim:
            self.bridge_h = nn.Linear(hid_dim, dec_hid, bias=True)
            self.bridge_c = nn.Linear(hid_dim, dec_hid, bias=True)
        else:
            self.bridge_h = None
            self.bridge_c = None

    def forward(self, src, source_mask):
        lengths = (~source_mask).sum(1).cpu()
        emb = self.drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        enc_outs, (h, c) = self.rnn(packed)

        #Needed for attention mechanism
        enc_outs, _ = nn.utils.rnn.pad_packed_sequence(enc_outs, batch_first=True, total_length=src.size(1))

        # I will use default hid_dim for decoder, so I don't need to bridge but I will keep the code for future use
        if self.bridge_h is not None:
            h0 = torch.tanh(self.bridge_h(h))
            c0 = torch.tanh(self.bridge_c(c))
        else:
            h0, c0 = h, c

        return enc_outs, (h0, c0)



class Decoder(nn.Module):

    def __init__(self, target_vocab_size, emb_dim, hid_dim,
                 num_layers, dropout, attn_type, pad_id=PAD_ID):
        super().__init__()
        self.emb = nn.Embedding(target_vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)

        # choose attention type
        if attn_type == "additive":
            self.attn = AdditiveAttention(enc_dim=hid_dim, dec_dim=hid_dim, attn_dim=hid_dim)
        elif attn_type == "multiplicative":
            self.attn = MultiplicativeAttention(enc_dim=hid_dim, dec_dim=hid_dim)
        elif attn_type == "scaled_dot":
            self.attn = ScaledDotProductAttention(enc_dim=hid_dim, dec_dim=hid_dim)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        # LSTM input = embedding + context
        self.rnn = nn.LSTM(input_size=emb_dim + hid_dim,
                           hidden_size=hid_dim,
                           num_layers=num_layers,
                           batch_first=True)
        # final linear layer
        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, target_vocab_size)

    def forward(self, target_input, target_mask, enc_outs, hidden0, source_mask):
        """
        target_input: (Batch_size, Sequence_length)
        enc_outs: (Batch_size, Sequence_length, hid_dim)
        hidden0: (h0, c0) where each is (num_layers, Batch_size, hid_dim)
        source_mask: (Batch_size, Sequence_length)
        """
        h, c = hidden0
        batch_size, sequence_length = target_input.shape
        outputs, attn_list = [], []

        # first input (<sos>)
        input_token = target_input[:, 0]

        for t in range(sequence_length):
            emb_t = self.drop(self.emb(input_token)).unsqueeze(1)  # (B,1,E)
            query = h[-1]  # take top layer
            context, attn = self.attn(query, enc_outs, source_mask)  # (B,H),(B,S)
            context = context.unsqueeze(1)  # (B,1,H)
            rnn_in = torch.cat([emb_t, context], dim=-1)
            out, (h, c) = self.rnn(rnn_in, (h, c))
            concat = torch.cat([out, context, emb_t], dim=-1)
            logit = self.fc_out(concat).squeeze(1)
            outputs.append(logit)
            attn_list.append(attn)
            if t + 1 < sequence_length:
                input_token = target_input[:, t + 1]

        outputs = torch.stack(outputs, dim=1)
        attn_weights = torch.stack(attn_list, dim=1)
        return outputs, (h, c), attn_weights


class Seq2Seq(nn.Module):
    """
    Simple LSTM encoder-decoder with pluggable attention.

    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 emb_dim=256, enc_hid=512, dec_hid=512,
                 num_layers=2, attn_type="additive", dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size,
                               emb_dim=emb_dim,
                               hid_dim=enc_hid,
                               num_layers=num_layers,
                               dec_hid=dec_hid,
                               dropout=dropout)

        self.decoder = Decoder(target_vocab_size=tgt_vocab_size,
                               emb_dim=emb_dim,
                               hid_dim=dec_hid,
                               num_layers=num_layers,
                               dropout=dropout,
                               attn_type=attn_type)

    def forward(self, batch):
        src = batch["src"]                 # (Batch_size, Sequence_length)
        target_input = batch["target_input"]         
        source_mask = batch["source_mask"]   
        target_mask = batch["target_mask"]   

        enc_outs, hidden0 = self.encoder(src, source_mask)
        outputs, _, attn = self.decoder(target_input, target_mask, enc_outs, hidden0, source_mask)
        return outputs, attn
