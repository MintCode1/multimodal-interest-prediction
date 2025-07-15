import torch
import numpy as np
import matplotlib.pyplot as plt

from models.video_encoder import VideoEncoder

def plot_feature_map(feature_map):
    # Assuming shape - (batch, channels, frames, H, W)
    batch, channels, frames, H, W = feature_map.shape
    mid_frame = frames // 2
    first_channel = feature_map[0, 0, mid_frame, :, :].cpu().detach().numpy()

    plt.imshow(first_channel, cmap='viridis')
    plt.title("Feature Map - First Channel, Middle Frame")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_enc = VideoEncoder().to(device)
    video_enc.eval()

    idx = 0  # video selection
    frames = np.load(f"data/features/frames_{idx}.npy")
    frames = frames.transpose(0, 3, 1, 2)
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    frames = frames.permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        fmap = video_enc.get_feature_map(frames)
        plot_feature_map(fmap)
