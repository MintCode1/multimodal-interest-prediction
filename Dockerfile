FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir torch torchvision torchaudio \
    transformers pandas fastapi uvicorn kafka-python \
    opencv-python librosa matplotlib scikit-learn streamlit

EXPOSE 8000

CMD ["uvicorn", "pipeline.inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
