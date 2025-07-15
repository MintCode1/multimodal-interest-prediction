import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd

from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from models.user_sequence_encoder import UserSequenceEncoder
from models.fusion_predictor import FusionPredictor
from utils.user_embedding import encode_user_history

def load_real_data(data_dir="data/features", device="cpu"):
    video_feats, audio_feats, text_feats, video_ids = [], [], [], []

    indices = [f.split("_")[1].split(".")[0] for f in os.listdir(data_dir) if f.startswith("frames_")]

    for idx in indices:
        frames = np.load(f"{data_dir}/frames_{idx}.npy")
        frames = frames.transpose(0, 3, 1, 2)
        frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

        audio = np.load(f"{data_dir}/audio_feats_{idx}.npy")
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

        text = torch.load(f"{data_dir}/text_ids_{idx}.pt").to(device)

        video_feats.append(frames)
        audio_feats.append(audio)
        text_feats.append(text)
        video_ids.append(int(idx))

    return video_feats, audio_feats, text_feats, video_ids

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_enc = VideoEncoder().to(device)
    audio_enc = AudioEncoder().to(device)
    text_enc = TextEncoder().to(device)
    user_enc = UserSequenceEncoder().to(device)
    predictor = FusionPredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(video_enc.parameters()) +
                           list(audio_enc.parameters()) +
                           list(text_enc.parameters()) +
                           list(user_enc.parameters()) +
                           list(predictor.parameters()), lr=1e-4)

    video_inputs, audio_inputs, text_inputs, video_ids = load_real_data(device=device)

    logs_df = pd.read_csv("data/user_logs.csv")

    for epoch in range(5):
        epoch_loss = 0
        for v, a, t, vid in zip(video_inputs, audio_inputs, text_inputs, video_ids):
            v = v.permute(0, 2, 1, 3, 4)

            scores = logs_df[logs_df["video_id"] == vid]["engagement_score"]
            score_value = scores.mean() if len(scores) > 0 else 0.5

            user_input = encode_user_history(logs_df, user_id=0).to(device)

            video_feat = video_enc(v)
            audio_feat = audio_enc(a)
            text_feat = text_enc(t)
            user_feat = user_enc(user_input)

            target = torch.tensor([[score_value]], dtype=torch.float32).to(device)

            output = predictor(video_feat, audio_feat, text_feat, user_feat)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(video_inputs):.4f}")

    # Save normal weights
    torch.save(predictor.state_dict(), "model_weights.pth")
    print("Model weights saved to model_weights.pth")

    # Save TorchScript traced model
    example_video_feat = torch.randn(1, 256).to(device)
    example_audio_feat = torch.randn(1, 256).to(device)
    example_text_feat = torch.randn(1, 256).to(device)
    example_user_feat = torch.randn(1, 256).to(device)

    traced_predictor = torch.jit.trace(predictor, (example_video_feat, example_audio_feat, example_text_feat, example_user_feat))
    torch.jit.save(traced_predictor, "predictor_traced.pt")
    print("TorchScript model saved to predictor_traced.pt")

if __name__ == "__main__":
    train()
