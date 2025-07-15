import requests
import json
import matplotlib.pyplot as plt

def rank_all_videos(num_videos):
    video_scores = []

    for idx in range(num_videos):
        response = requests.post("http://127.0.0.1:8000/predict/", json={"index": idx})
        if response.status_code == 200:
            prob = response.json()["engagement_probability"]
            video_scores.append((idx, prob))
            print(f"Video {idx}: {prob:.4f}")
        else:
            print(f"Failed for video {idx}")

    # Sort by probability descending
    ranked = sorted(video_scores, key=lambda x: x[1], reverse=True)

    print("\n=== Top Videos ===")
    for rank, (idx, score) in enumerate(ranked[:10], start=1):
        print(f"{rank}. Video {idx} — Predicted Probability: {score:.4f}")

    # Save to file
    with open("ranked_videos.json", "w") as f:
        json.dump(ranked, f, indent=2)
    print("\nSaved ranked list to ranked_videos.json")

    # Plot top 10 as bar chart
    top_idxs = [f"Vid {i[0]}" for i in ranked[:10]]
    top_scores = [i[1] for i in ranked[:10]]

    plt.figure(figsize=(10, 6))
    plt.barh(top_idxs[::-1], top_scores[::-1], color="skyblue")
    plt.xlabel("Predicted Engagement Probability")
    plt.title("Top 10 Videos Likely to Go Viral")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # video count
    num_videos = 250
    rank_all_videos(num_videos)
