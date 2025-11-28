import math
import torch
import torch.nn as nn

def masked_softmax(scores, mask):  # scores: (Batch,Sequence) or (Batch,1,Sequence)
    if scores.dim() == 2:
        scores = scores.masked_fill(mask, float('-inf'))
        return torch.softmax(scores, dim=-1)
    elif scores.dim() == 3:
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        return torch.softmax(scores, dim=-1)
    else:
        raise ValueError("scores shape not supported")

class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, enc_outs, source_mask):

        # query: (Batch, dec_dim); enc_outs: (Batch,Sequence,enc_dim)
        h_proj = self.W_h(enc_outs)                     # (Batch,Sequence,attn_dim)
        s_proj = self.W_s(query).unsqueeze(1)           # (Batch,1,attn_dim)
        scores = self.v(torch.tanh(h_proj + s_proj)).squeeze(-1)  # (Batch,Sequence)
        attn = masked_softmax(scores, source_mask)     # (Batch,Sequence)
        context = torch.bmm(attn.unsqueeze(1), enc_outs).squeeze(1)  # (Batch,enc_dim)
        return context, attn

class MultiplicativeAttention(nn.Module):
    
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W = nn.Linear(enc_dim, dec_dim, bias=False)

    def forward(self, query, enc_outs, source_mask):
        proj = self.W(enc_outs)                         # (Batch,Sequence,dec_dim)
        scores = torch.bmm(proj, query.unsqueeze(-1)).squeeze(-1)  # (Batch,Sequence)
        attn = masked_softmax(scores, source_mask)     # (Batch,Sequence)
        context = torch.bmm(attn.unsqueeze(1), enc_outs).squeeze(1)  # (Batch,enc_dim)
        return context, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, d_k=None, d_v=None):
        super().__init__()
        d_k = d_k or dec_dim
        d_v = d_v or enc_dim
        self.Wq = nn.Linear(dec_dim, d_k, bias=False)
        self.Wk = nn.Linear(enc_dim, d_k, bias=False)
        self.Wv = nn.Linear(enc_dim, d_v, bias=False)
        self.scale = math.sqrt(d_k)

    def forward(self, query, enc_outs, source_mask):
        # query: (Batch, dec_dim); enc_outs: (Batch,Sequence,enc_dim)
        Q = self.Wq(query).unsqueeze(1)                # (Batch,1,d_k)
        K = self.Wk(enc_outs)                          # (Batch,Sequence,d_k)
        V = self.Wv(enc_outs)                          # (Batch,Sequence,d_v)
        scores = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) / self.scale  # (Batch,Sequence)
        attn = masked_softmax(scores, source_mask)    # (Batch,Sequence)
        context = torch.bmm(attn.unsqueeze(1), V).squeeze(1)  # (Batch,d_v)
        return context, attn
