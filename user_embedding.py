import torch
import torch.nn as nn

def encode_user_history(log_df, user_id, embed_dim=256):
    """
    Create a user sequence embedding based on user's action history using GRU.
    """
    action_to_idx = {"view": 0, "like": 1, "skip": 2, "pause": 3, "rewind": 4}

    # Get sorted actions for current user
    user_logs = log_df[log_df["user_id"] == user_id].sort_values(by="timestamp")
    action_seq = [action_to_idx.get(a, 0) for a in user_logs["action"].tolist()]

    if len(action_seq) == 0:
        # No logs for user then return random embedding
        return torch.randn(10, 1, embed_dim)

    seq_tensor = torch.tensor(action_seq).unsqueeze(1)  # (seq_len, batch=1)

    embed = nn.Embedding(len(action_to_idx), embed_dim)
    gru = nn.GRU(embed_dim, embed_dim)

    embedded = embed(seq_tensor)
    _, hidden = gru(embedded)

    # Repeat hidden state for sequence
    repeated = hidden.repeat(10, 1, 1)  # (seq_len, batch, embed_dim)
    return repeated
