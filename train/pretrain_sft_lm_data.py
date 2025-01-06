import os
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast
import ml_utils
import datasets
import ml_utils.data
import sequence_modelling




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
        self.user_start_tokens = tokenizer.encode("<|user|>", return_tensors="pt")[0]
        self.system_start_tokens = tokenizer.encode("<|assistant|>", return_tensors="pt")[0]
        self.eos_token = tokenizer.encode("</s>", return_tensors="pt")[0]

    def prepare_data(self):
        that = self
        class Transform(ml_utils.data.LiftedTransform):

            def transform(self, messages):
                text = sequence_modelling.apply_chat_template(messages)
                tokenized = sequence_modelling.tokenize_encode_pad_max_len(
                    that.tokenizer, that.max_seq_len, text
                )
                labels = sequence_modelling.labels_skip_user_prompts(
                    that.system_start_tokens, that.eos_token, tokenized["input_ids"]
                )
                return {"input_ids": tokenized["input_ids"], "labels": labels, "attn_mask": tokenized["attn_mask"]}

        transform = Transform()
        print(transform.transform)
        if not os.path.exists("train_sft"):
            train_dataset = datasets.load_dataset(self.path, split="train_sft")
            train_dataset = train_dataset.map(transform).select_columns(["input_ids", "labels", "attn_mask"])
            train_dataset.save_to_disk("train_sft")
        if not os.path.exists("val_sft"):
            eval_dataset = datasets.load_dataset(self.path, split="test_sft")
            eval_dataset = eval_dataset.map(transform).select_columns(["input_ids", "labels", "attn_mask"])
            eval_dataset.save_to_disk("val_sft")

    def setup(self, stage: str = "fit"):
        print("setup")
        if stage == "fit":
            self.train_dataset = datasets.load_from_disk( "train_sft", )
            self.train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
            self.val_dataset = datasets.load_from_disk( "val_sft", )
            self.val_dataset.set_format(type="torch", columns=["input_ids", "labels"])

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
    import sampler.lm_tokenizer as lm_tokenizer
    tokenizer = lm_tokenizer.load_tokenizer()
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
