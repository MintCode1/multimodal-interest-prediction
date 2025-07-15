import torch
import numpy as np
import os
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.video_encoder import VideoEncoder

def load_video_embeddings(data_dir="data/features", device="cpu"):
    embeddings = []
    video_indices = []
    indices = [f.split("_")[1].split(".")[0] for f in os.listdir(data_dir) if f.startswith("frames_")]
    video_enc = VideoEncoder().to(device)
    video_enc.eval()

    with torch.no_grad():
        for idx in indices:
            frames = np.load(f"{data_dir}/frames_{idx}.npy")
            frames = frames.transpose(0, 3, 1, 2)
            frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
            frames = frames.permute(0, 2, 1, 3, 4)

            feat = video_enc(frames)
            embeddings.append(feat.cpu().numpy().flatten())
            video_indices.append(int(idx))

    return np.array(embeddings), video_indices

def plot_tsne(embeddings, categories):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=3000, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    # Create color mapping for categories
    unique_cats = list(set(categories.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_cats)}

    plt.figure(figsize=(10, 8))
    for idx, (x, y) in enumerate(reduced):
        cat = categories[str(idx)]
        plt.scatter(x, y, color=color_map[cat], label=cat, alpha=0.7, edgecolor='k', s=50)

    # Create legend without duplicates
    handles = []
    labels = []
    for cat, color in color_map.items():
        handles.append(plt.Line2D([], [], marker="o", color='w', markerfacecolor=color, markersize=10))
        labels.append(cat)
    plt.legend(handles, labels, loc='best')

    plt.title("t-SNE Visualization of Video Embeddings (Colored by Category)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, video_indices = load_video_embeddings(device=device)

    # Load categories
    with open("data/features/categories.json") as f:
        categories = json.load(f)

    plot_tsne(embeddings, categories)
