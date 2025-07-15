import torch
import torch.nn as nn

class UserSequenceEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(UserSequenceEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4), num_layers=2
        )

    def forward(self, x):
        # x shape: (seq_len, batch, embed_dim)
        x = self.transformer(x)
        return x.mean(dim=0)  # return pooled representation
