import torch
from transformers import PreTrainedTokenizerFast

def load_tokenizer():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='trained_tokenizers/v1.json')
    tokenizer.unk_token_id = 0
    tokenizer.sep_token_id = 1
    tokenizer.pad_token_id = 2
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    return tokenizer

def data_collator(tokenizer, max_seq_len, batch):
    strs = [x["text"] for x in batch]
    encoding = tokenizer(strs, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
    input_ids = encoding["input_ids"]
    
    # Shift labels by one position to the right
    labels = input_ids.clone()
    labels = torch.cat([labels[:, 1:], labels[:, :1]], dim=1)  # Shift left
    labels[labels == tokenizer.pad_token_id] = -100
    
    token_count = (input_ids != tokenizer.pad_token_id).sum().item()
    return {"input_ids": input_ids, "labels": labels, "token_count": token_count}

