import torch
import torch.nn as nn

class FusionPredictor(nn.Module):
    def __init__(self, embed_dim=256):
        super(FusionPredictor, self).__init__()
        self.fc1 = nn.Linear(embed_dim * 4, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, video_feat, audio_feat, text_feat, user_feat):
        x = torch.cat([video_feat, audio_feat, text_feat, user_feat], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
