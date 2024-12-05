from typing import List, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

import json
import micro_lm_model
import dataclasses

@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    generation_max_length: int = 100
    temperature: float = 1.0 
    top_p: float = 0.8
    repetition_penalty: float = 1.0

    def from_json(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
            return GenerationConfig(**data)


class RepetitionPenaltyLogitsProcessor():
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores

class ChatWrapper(nn.Module):

    def __init__(
        self,
        model: micro_lm_model.TransformerDecoder,
        tokenizer: PreTrainedTokenizerFast,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device

    def forward(self, input_ids, attention_mask, input_pos):
        return self.model(input_ids, attention_mask=attention_mask, input_pos=input_pos)

    def show_output_probs(self, text: str):
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        logits = self.model(input_ids)[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        top_10 = torch.topk(probs, 10)
        result = []
        for (token, prob) in zip(top_10.indices, top_10.values):
            result.append((self.tokenizer.decode(token), prob))
        return result

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            
            Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Replace logits to be removed with -inf in the sorted_logits
        sorted_logits[sorted_indices_to_remove] = filter_value
        # Then reverse the sorting process by mapping back sorted_logits to their original position
        logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

        pred_token = torch.multinomial(F.softmax(logits, -1), 1) # [BATCH_SIZE, 1]
        return pred_token

    def generate_ids(
        self, input_ids: torch.LongTensor, config: GenerationConfig = None
    ):
        if config is None:
            config = GenerationConfig()
        position_ids = torch.arange(0, input_ids.size(1),)

        self.model.eval()
        batch_size = input_ids.size(0)
        self.model.setup_caches(batch_size, self.dtype, self.device)
        output = [[] for _ in range(input_ids.size(0))]
        input = input_ids.to(torch.long)
        ith_token = input_ids.size(1)
        for i in range(config.generation_max_length):
            logits = self.model(input, input_pos=position_ids)
            next_token_logits = logits[:, -1, :] 
            if config.temperature == 0.0:
                next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                next_token_ids = self.top_k_top_p_filtering(next_token_logits / config.temperature, top_k=0, top_p=config.top_p)
            for i, token_id in enumerate(next_token_ids):
                output[i].append(token_id.item())

            input = next_token_ids
            position_ids = torch.ones(1, dtype=torch.long) * ith_token
            ith_token += 1
            if torch.all(next_token_ids == self.tokenizer.eos_token_id):
                break
        self.model.remove_caches()
        self.model.train()
        return torch.tensor(output)

    @torch.no_grad()
    def generate(self, text: Union[str, List[str]], config: GenerationConfig = None):
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        output = self.generate_ids(input_ids, config)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
