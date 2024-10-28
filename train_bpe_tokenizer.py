import argparse
import os
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from datasets import load_dataset

def bpe_tokenization(model_root, data):
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=True))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(" ", behavior="merged_with_next"),
        pre_tokenizers.Split(Regex('\d|[\u2E80-\u2FDF\u3040-\u318F\u31A0-\u31BF\u31F0-\u31FF\u3400-\u4DB5\u4E00-\u9FFF\uA960-\uA97F\uAC00-\uD7FF]'), behavior='isolated'),
        pre_tokenizers.Split(Regex(' *(\w+|[^\w\s]+)'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"], vocab_size=32000,
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.save(model_root)
    validate(model_root)

def validate(model_root):
    tokenizer = Tokenizer.from_file(model_root)
    output = tokenizer.encode("Hello, y'all! How are you 😁? 你好！ 今天吃了啥 1234 1 2 3 4.")
    print(output.tokens)
    decoded = tokenizer.decode(output.ids)
    print(decoded)

def main(model_root, dataset_root):
    dataset = load_dataset(dataset_root)
    data = dataset["train"]["text"]
    bpe_tokenization(model_root, data)
    validate(model_root)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=str, default="tokenizer.json")
    parser.add_argument("--dataset_root", type=str, default="DKYoon/SlimPajama-6B")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.model_root, args.dataset_root)
