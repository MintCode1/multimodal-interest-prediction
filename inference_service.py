from fastapi import FastAPI, Request
import torch
import numpy as np
import pandas as pd
import random
import json

from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from models.user_sequence_encoder import UserSequenceEncoder
from models.fusion_predictor import FusionPredictor
from utils.user_embedding import encode_user_history

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

video_enc = VideoEncoder().to(device)
audio_enc = AudioEncoder().to(device)
text_enc = TextEncoder().to(device)
user_enc = UserSequenceEncoder().to(device)

# Load two versions of predictor
predictor_a = torch.jit.load("predictor_traced.pt", map_location=device).to(device)  # With user embeddings
predictor_b = torch.jit.load("predictor_traced.pt", map_location=device).to(device)  # Same model but ignoring user embeddings

predictor_a.eval()
predictor_b.eval()

log_df = pd.read_csv("data/user_logs.csv")

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    idx = int(data.get("index", 0))

    # Pick A or B group
    bucket = random.choice(["A", "B"])

    frames = np.load(f"data/features/frames_{idx}.npy")
    frames = frames.transpose(0, 3, 1, 2)
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    frames = frames.permute(0, 2, 1, 3, 4)

    audio_feats = np.load(f"data/features/audio_feats_{idx}.npy")
    audio_feats = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(device)

    text_ids = torch.load(f"data/features/text_ids_{idx}.pt").to(device)

    # Generate user embedding only for A
    if bucket == "A":
        user_tensor = encode_user_history(log_df, user_id=0).to(device)
    else:
        user_tensor = torch.zeros(10, 1, 256).to(device)  # Empty embedding for baseline

    video_feat = video_enc(frames)
    audio_feat = audio_enc(audio_feats)
    text_feat = text_enc(text_ids)
    user_feat = user_enc(user_tensor)

    if bucket == "A":
        prob = predictor_a(video_feat, audio_feat, text_feat, user_feat).item()
    else:
        prob = predictor_b(video_feat, audio_feat, text_feat, user_feat).item()

    # Log request
    log_entry = {
        "bucket": bucket,
        "index": idx,
        "predicted_prob": prob
    }
    with open("ab_test_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"bucket": bucket, "engagement_probability": prob}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
