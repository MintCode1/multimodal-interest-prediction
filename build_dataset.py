import os
import subprocess
import json
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import numpy as np
import torch
from preprocess_video import extract_frames
from preprocess_audio import extract_audio_features
from preprocess_text import text_to_ids

# Creating folders
os.makedirs("data/videos", exist_ok=True)
os.makedirs("data/audios", exist_ok=True)
os.makedirs("data/features", exist_ok=True)

# Playlist categories
playlists = {
    "Pop": "https://www.youtube.com/playlist?list=PLGBuKfnErZlB3AThAEKz8_3kbYTocgfbB",
    "HipHop": "https://www.youtube.com/playlist?list=PLoXpL6eMBpFWuFT1pJnluYIu_xeZjVxQ0",
    "Rock": "https://www.youtube.com/playlist?list=PLVQ7g3e6O27cH8KG9mktLWH8zcqiwTntP",
    "Christmas": "https://www.youtube.com/playlist?list=PLyORnIW1xT6zGkZuHOleCkyUdTIlSQkS9",
    "Jazz": "https://www.youtube.com/playlist?list=PL8F6B0753B2CCA128"
}

ydl_opts = {
    "extract_flat": True,
    "quiet": True,
    "skip_download": True,
}

video_counter = 0

for category, playlist_url in playlists.items():
    print(f"\nProcessing category: {category}")

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        entries = info.get("entries", [])

    video_urls = ["https://www.youtube.com/watch?v=" + entry["id"] for entry in entries if "id" in entry]
    video_urls = video_urls[:50]  # num per playlist

    for idx, url in enumerate(video_urls):
        video_filename = f"video_{video_counter}.mp4"
        video_output_path = f"data/videos/{video_filename}"
        audio_output_path = f"data/audios/audio_{video_counter}.wav"

        caption = f"{category} music video #{idx}"

        try:
            print(f"\n[{category}] Downloading video {idx + 1}/{len(video_urls)}: {url}")

            subprocess.run(["yt-dlp", "-f", "best", "-o", video_output_path, url], check=True)
            print(f"Downloaded to: {video_output_path}")

            # Extract frames
            frames = extract_frames(video_output_path)
            np.save(f"data/features/frames_{video_counter}.npy", frames)
            print(f"Saved frames to: data/features/frames_{video_counter}.npy")

            # Extract and save audio
            audio = AudioSegment.from_file(video_output_path)
            audio.export(audio_output_path, format="wav")
            print(f"Exported audio to: {audio_output_path}")

            # Extract audio features
            audio_feats = extract_audio_features(audio_output_path)
            np.save(f"data/features/audio_feats_{video_counter}.npy", audio_feats)
            print(f"Saved audio features to: data/features/audio_feats_{video_counter}.npy")

            # Process text caption
            token_ids = text_to_ids(caption)
            torch.save(token_ids, f"data/features/text_ids_{video_counter}.pt")
            print(f"Saved text tokens to: data/features/text_ids_{video_counter}.pt")

            video_counter += 1

        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

print("\nAll playlists processed and features saved!")
