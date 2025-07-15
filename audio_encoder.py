import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)
