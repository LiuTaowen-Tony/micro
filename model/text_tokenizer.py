from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast
import ml_utils.misc as misc

@dataclass
class TrainedTextTokenizerConfig:
    tokenizer_file: str = misc.relative_path(
        "../trained_models/text_tokenizer_v1/v1.json", __file__
    )

    def build_model(self):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_file)
        return tokenizer

@dataclass
class TextTokenizerConfig:
    tokenizer_file: str = misc.relative_path(
        "../trained_models/text_tokenizer_v1/v1.json", __file__
    )
    vocab_size: int = 32000
    unk_token_id: int = 0
    sep_token_id: int = 1
    pad_token_id: int = 2
    eos_token_id: int = 2
    padding_side: str = "left"

    def build_model(self):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_file)
        assert tokenizer.vocab_size == self.vocab_size, "Vocab size does not match"
        tokenizer.unk_token_id = self.unk_token_id
        tokenizer.sep_token_id = self.sep_token_id
        tokenizer.pad_token_id = self.pad_token_id
        tokenizer.eos_token_id = self.eos_token_id
        tokenizer.padding_side = self.padding_side

        return tokenizer


def load_tokenizer():
    return TextTokenizerConfig().build_model()
    # file_path = misc.relative_path("../trained_models/text_tokenizer_v1/v1.json", __file__)
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
    # tokenizer.unk_token_id = 0
    # tokenizer.sep_token_id = 1
    # tokenizer.pad_token_id = 2
    # tokenizer.eos_token_id = 2
    # tokenizer.padding_side = "left"

    # return tokenizer
