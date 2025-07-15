import random
import pandas as pd

def simulate_user_logs(num_users=1000, num_videos=500, max_events=50):
    logs = []
    for user_id in range(num_users):
        for _ in range(random.randint(10, max_events)):
            video_id = random.randint(0, num_videos)
            action = random.choice(["view", "like", "skip", "pause", "rewind"])
            timestamp = random.randint(1_600_000_000, 1_700_000_000)

            # Simulate engagement score
            score = 0.8 if action == "like" else (0.2 if action == "skip" else 0.5)

            logs.append({
                "user_id": user_id,
                "video_id": video_id,
                "action": action,
                "timestamp": timestamp,
                "engagement_score": score
            })

    df = pd.DataFrame(logs)
    df.to_csv("data/user_logs.csv", index=False)
    print("User logs simulated and saved with engagement scores.")

if __name__ == "__main__":
    simulate_user_logs()
