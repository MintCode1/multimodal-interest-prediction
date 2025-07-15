import pandas as pd
import json

def load_ab_logs(file_path="ab_test_logs.jsonl"):
    with open(file_path) as f:
        logs = [json.loads(line.strip()) for line in f.readlines()]
    return pd.DataFrame(logs)

def analyze_logs(df):
    print("\n=== A/B Test Analysis ===")
    print(df.groupby("bucket")["predicted_prob"].describe())

if __name__ == "__main__":
    df = load_ab_logs()
    analyze_logs(df)
