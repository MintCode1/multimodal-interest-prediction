def rank_videos(predicted_probs, video_list):
    ranked = sorted(zip(video_list, predicted_probs), key=lambda x: x[1], reverse=True)
    return [v for v, _ in ranked]
