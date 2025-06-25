import torch
import torch.nn as nn
from config import Config

def generate_causal_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(Config.vocab_size, Config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, Config.seq_len, Config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.d_model,
            nhead=Config.n_heads,
            dim_feedforward=4 * Config.d_model,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.n_layers)
        self.ln_f = nn.LayerNorm(Config.d_model)
        self.head = nn.Linear(Config.d_model, Config.vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok = self.token_emb(idx)             # (B, T, d_model)
        pos = self.pos_emb[:, :T, :]         # (1, T, d_model)
        x = tok + pos
        x = x.transpose(0, 1)                # (T, B, d_model)
        mask = generate_causal_mask(T).to(x.device)
        x = self.transformer(x, mask=mask)
        x = x.transpose(0, 1)                # (B, T, d_model)
        x = self.ln_f(x)
        logits = self.head(x)                # (B, T, vocab_size)
        return logits
