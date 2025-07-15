from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def text_to_ids(text, max_len=20):
    tokens = tokenizer.encode_plus(text, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")
    return tokens['input_ids']
