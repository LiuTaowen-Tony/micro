from transformers import PreTrainedTokenizerFast
import ml_utils.misc as misc

def load_tokenizer():
    file_path = misc.relative_path("trained_models/text_tokenizer_v1/v1.json", __file__)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
    tokenizer.unk_token_id = 0
    tokenizer.sep_token_id = 1
    tokenizer.pad_token_id = 2
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    return tokenizer
