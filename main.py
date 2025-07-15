import subprocess

def main():
    print("1️⃣ Simulating user logs...")
    subprocess.run(["python", "data/simulate_user_logs.py"])

    print("2️⃣ Training multimodal model...")
    subprocess.run(["python", "training/train_multimodal.py"])

    print("3️⃣ Starting streaming consumer (requires Kafka running)...")
    # subprocess.run(["python", "pipeline/streaming_consumer.py"])

    print("4️⃣ To start FastAPI inference service, run: python pipeline/inference_service.py")

if __name__ == "__main__":
    main()
