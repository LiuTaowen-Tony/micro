from transformers import PreTrainedTokenizerFast
from ml_utils import path

def load_tokenizer():
    file_path = path.relative_path("trained_tokenizers/v1.json", __file__)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
    tokenizer.unk_token_id = 0
    tokenizer.sep_token_id = 1
    tokenizer.pad_token_id = 2
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    return tokenizer
