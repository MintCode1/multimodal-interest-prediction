import shutil
import os

def cleanup_videos_and_audios():
    # Delete videos
    if os.path.exists("data/videos"):
        shutil.rmtree("data/videos")
        os.makedirs("data/videos", exist_ok=True)
        print("Deleted all videos and recreated empty videos folder.")
    else:
        print("videos folder not found. Skipping.")

    # Delete audios
    if os.path.exists("data/audios"):
        shutil.rmtree("data/audios")
        os.makedirs("data/audios", exist_ok=True)
        print("Deleted all audios and recreated empty audios folder.")
    else:
        print("audios folder not found. Skipping.")

if __name__ == "__main__":
    cleanup_videos_and_audios()
