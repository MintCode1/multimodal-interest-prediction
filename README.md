# Multimodal Real-Time User Interest Prediction

## Overview
This project predicts user engagement probability on videos using multimodal signals:
- Video frames (3D CNN)
- Audio (MFCC features)
- Text captions (BERT embeddings)
- User interaction logs (Transformer-based embeddings)

Designed to simulate real-world recommendation and personalization systems

## Features
- Downloads and preprocesses real YouTube videos
- Builds user embeddings from simulated logs
- PyTorch-based multimodal deep learning model
- Real-time FastAPI serving
- Streamlit dashboard for live demo

## Project Structure
multimodal_interest_prediction/
├── data/

│   ├── build_dataset.py          # Download & extract video/audio/text features

│   ├── preprocess_video.py       # Frame extraction logic

│   ├── preprocess_audio.py       # Audio feature extraction logic

│   ├── preprocess_text.py        # Text preprocessing logic
│   ├── simulate_user_logs.py     # Generates simulated user interaction logs
│   ├── videos/                   # Downloaded raw videos
│   ├── audios/                   # Extracted audio files
│   └── features/                 # Saved feature files (frames, audio_feats, text_ids)
│
├── models/
│   ├── video_encoder.py
│   ├── audio_encoder.py
│   ├── text_encoder.py
│   ├── user_sequence_encoder.py
│   └── fusion_predictor.py
│
├── utils/
│   └── user_embedding.py        # Generates user embeddings from logs
│
├── training/
│   └── train_multimodal.py      # Main training script using real data
│
├── pipeline/
│   ├── inference_service.py     # FastAPI serving endpoint using real features and logs
│   └── streaming_consumer.py    # Kafka consumer for real-time logs
│
├── visualization/
│   ├── dashboard.py             # Streamlit UI for live predictions
│   ├── visualize_embeddings.py  # t-SNE visualization
│   └── visualize_feature_maps.py # Feature map visualization
│   └── analyze_ab_logs.py       # A/B logs visualization
│
├── Dockerfile                   # Containerization support (optional)
├── docker-compose.yml           # Docker setup with Kafka (optional)
├── requirements.txt            # Python dependencies
├── README.md                   
└── main.py                     

## Possible Improvements
- Use advanced pretrained video encoders
- Improve user embeddings with sequential log modeling
- Deploy on cloud with scalable serving (e.g., TorchServe)
- Add visualizations (e.g., t-SNE)
- Add explainability (e.g., Grad-CAM)

## License
MIT — free to use and modify.
