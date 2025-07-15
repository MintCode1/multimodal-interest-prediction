import torch
import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, x):
        outputs = self.bert(input_ids=x, attention_mask=(x != 0))
        pooled = outputs.pooler_output
        return self.fc(pooled)
