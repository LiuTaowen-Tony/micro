import os
import numpy as np
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


class SFTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int,
        max_seq_len: int,
        path: str,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_dataset = None
        self.val_dataset = None
        self.path = path
        self.user_start_tokens = tokenizer.encode("<|user|>")
        self.system_start_tokens = tokenizer.encode("<|assistant|>")
        self.eos_token = tokenizer.encode("</s>")

    def prepare_data(self):
        if not os.path.exists("train_sft"):
            train_dataset = datasets.load_dataset(self.path, split="train_sft")
            train_dataset = train_dataset.map(self.get_input_ids_labels,).select_columns(
                ["input_ids", "labels"]
            )
            train_dataset.save_to_disk("train_sft")
        if not os.path.exists("val_sft"):
            eval_dataset = datasets.load_dataset(self.path, split="test_sft")
            eval_dataset = eval_dataset.map(self.get_input_ids_labels,).select_columns(
                ["input_ids", "labels"]
            )
            eval_dataset.save_to_disk("val_sft")

    def setup(self, stage: str = "fit"):
        print("setup")
        if stage == "fit":
            self.train_dataset = datasets.load_from_disk( "train_sft", )
            self.train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
            self.val_dataset = datasets.load_from_disk( "val_sft", )
            self.val_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    def apply_chat_template(self, messages):
        """
        Convert messages to a formatted text string
        """
        text = ""
        for message in messages:
            if message["role"] == "user":
                text += f"<|user|>{message['content']}</s>"
            elif message["role"] == "assistant":
                text += f"<|assistant|>{message['content']}</s>"
        return text

    def find_token_sequence(self, input_ids, token_sequence):
        """
        Find the starting indices of a specific token sequence in input_ids

        :param input_ids: Input token ids tensor
        :param token_sequence: Sequence of tokens to find
        :return: Tensor of starting indices
        """
        seq_len = len(token_sequence)
        matches = []

        for i in range(input_ids.shape[1] - seq_len + 1):
            if torch.equal(input_ids[0, i : i + seq_len], torch.tensor(token_sequence)):
                matches.append(i)

        return torch.tensor(matches, dtype=torch.long)

    def get_input_ids_labels(self, item):
        """
        Process input messages to create input_ids and labels for training

        :param item: Dictionary containing 'messages' list
        :return: Dictionary with input_ids and labels
        """
        # Encode the entire conversation
        messages = item["messages"]
        text = self.apply_chat_template(messages)

        # Encode with truncation and padding
        input_ids = self.tokenizer.encode(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
            padding_side="left",
        )
        # Initialize labels with -100 (ignore tokens)
        labels = torch.full_like(input_ids, -100)

        # Find system response start positions
        system_start_positions = self.find_token_sequence(
            input_ids, self.system_start_tokens
        )
        eos_positions = self.find_token_sequence(input_ids, self.eos_token)

        # Process each system response
        for start_pos in system_start_positions:
            # Find the end of this system response
            end_pos = start_pos + len(self.system_start_tokens)

            # Look for next user start or end of sequence
            next_end_pos = min(
                eos_positions[eos_positions > end_pos], default=input_ids.shape[1]
            )
            if next_end_pos == input_ids.shape[1]:
                next_end_pos = input_ids.shape[1] - 1
            labels[0, end_pos - 1 : next_end_pos - 1] = input_ids[
                0, end_pos:next_end_pos
            ]

        assert input_ids.shape == labels.shape
        assert input_ids.shape[1] == self.max_seq_len
        return {"input_ids": input_ids[0], "labels": labels[0]}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=9
        )

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=9
        )
        print(dataloader)
        print(dataloader.dataset)
        return dataloader

    def state_dict(self):
        try:
            return {
                "train_dataset": self.train_dataset.state_dict(),
                "val_dataset": self.val_dataset.state_dict(),
            }
        except:
            return {}

    def load_state_dict(self, state_dict):
        try:
            self.train_dataset.load_state_dict(state_dict["train_dataset"])
            self.val_dataset.load_state_dict(state_dict["val_dataset"])
        except:
            pass


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
    # data_module = FillSeqDataModule(tokenizer, 8, 512, "DKYoon/SlimPajama-6B")
    # data_module.setup()
    # train_loader = data_module.train_dataloader()
    # for i, x in zip(range(5), train_loader):
    #     print(x)
    # state_dict = data_module.state_dict()
    # for i, x in zip(range(5), train_loader):
    #     print(x)
    # data_module.load_state_dict(state_dict)
    # for i, x in zip(range(5), train_loader):
    #     print(x)
    # print("DataModule state dict test passed")

    data_module = SFTDataModule(tokenizer, 8, 1024, "HuggingFaceH4/ultrachat_200k")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for i, x in zip(range(5), train_loader):
        # print(x)
        pass
    state_dict = data_module.state_dict()
    for i, x in zip(range(5), train_loader):
        print(x)
    data_module.load_state_dict(state_dict)
    for i, x in zip(range(5), train_loader):
        print(x)
    print("DataModule state dict test passed")
