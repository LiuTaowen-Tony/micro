import torch
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast
from ml_utils import path
import ml_utils
import datasets

import ml_utils.data


def load_tokenizer():
    file_path = path.relative_path("trained_tokenizers/v1.json", __file__)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
    tokenizer.unk_token_id = 0
    tokenizer.sep_token_id = 1
    tokenizer.pad_token_id = 2
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    return tokenizer


def data_collator(tokenizer, max_seq_len, batch):
    strs = [x["text"] for x in batch]
    encoding = tokenizer(strs, truncation=True, padding='max_length',
                         max_length=max_seq_len, return_tensors='pt')
    input_ids = encoding["input_ids"]

    # Shift labels by one position to the right
    labels = input_ids.clone()
    labels = torch.cat([labels[:, 1:], labels[:, :1]], dim=1)  # Shift left
    labels[labels == tokenizer.pad_token_id] = -100

    token_count = (input_ids != tokenizer.pad_token_id).sum().item()
    return {"input_ids": input_ids, "labels": labels, "token_count": token_count}


class FillSeqDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: datasets.IterableDataset, tokenizer: PreTrainedTokenizerFast, max_seq_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.buffer = []
        self.word_count = 0
        self.max_seq_len = max_seq_len

    def __iter__(self):
        for x in self.dataset:
            text = x["text"]
            encoding = self.tokenizer.encode(text, return_tensors='pt')[0]
            self.word_count = self.word_count + len(encoding)
            self.buffer.append(encoding)
            if self.word_count > self.max_seq_len:
                input_ids = torch.cat(self.buffer)[:self.max_seq_len]
                labels = input_ids.clone()
                labels[:-1] = input_ids[1:]
                labels[-1] = -100
                yield {"input_ids": input_ids, "labels": labels}
                self.word_count = 0
                self.buffer = []

    def state_dict(self):
        return {"dataset_state_dict": self.dataset.state_dict(),
                "word_count": self.word_count,
                "buffer": self.buffer}

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict["dataset_state_dict"])
        self.word_count = state_dict["word_count"]
        self.buffer = state_dict["buffer"]


class FillSeqDataModule(pl.LightningDataModule):
    # use prepare data can avoid dataset being loaded by multiple times
    def __init__(self, tokenizer: PreTrainedTokenizerFast, batch_size: int, max_seq_len: int, path: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_dataset = None
        self.val_dataset = None
        self.path = path

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            train_raw_data = datasets.load_dataset(
                self.path, split="train", streaming=True)
            val_raw_data = datasets.load_dataset(
                self.path, split="validation", streaming=True)
            val_raw_data = ml_utils.data.IterableSubset(val_raw_data, 1000)
            self.train_dataset = FillSeqDataset(
                train_raw_data, self.tokenizer, self.max_seq_len)
            self.val_dataset = FillSeqDataset(
                val_raw_data, self.tokenizer, self.max_seq_len)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=9)

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=9)
        print(dataloader)
        print(dataloader.dataset)
        return dataloader

    def state_dict(self):
        return {"train_dataset": self.train_dataset.state_dict(),
                "val_dataset": self.val_dataset.state_dict()}

    def load_state_dict(self, state_dict):
        self.train_dataset.load_state_dict(state_dict["train_dataset"])
        self.val_dataset.load_state_dict(state_dict["val_dataset"])


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    data_module = FillSeqDataModule(tokenizer, 8, 512, "DKYoon/SlimPajama-6B")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for i, x in zip(range(5), train_loader):
        print(x)
    state_dict = data_module.state_dict()
    for i, x in zip(range(5), train_loader):
        print(x)
    data_module.load_state_dict(state_dict)
    for i, x in zip(range(5), train_loader):
        print(x)
    print("DataModule state dict test passed")
