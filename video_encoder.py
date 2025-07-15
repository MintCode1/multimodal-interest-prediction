import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(VideoEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature_map(self, x):
        """
        Return the intermediate feature map before pooling.
        """
        for layer in self.cnn[:-1]:  # Skip the final AdaptiveAvgPool3d
            x = layer(x)
        return x
