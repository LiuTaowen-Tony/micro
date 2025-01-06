from torch import nn
import transformers
import dataclasses
from torch import FloatTensor, LongTensor, BoolTensor
from typing import Optional

@dataclasses.dataclass
class TransformerAdapterConfig:
    model_name: str = "distilbert-base-uncased"
    hidden_dim: int = 768

    def build_model(self):
        return TransformerAdapter(self, 
                                  transformers.AutoModel.from_pretrained(self.model_name)
                                  )


class TransformerAdapter(nn.Module):
    def __init__(self, config: TransformerAdapterConfig, model=nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)

    def forward(self, x: LongTensor, *, mask: Optional[BoolTensor] = None):
        last_hidden_state = self.model(x, attention_mask=mask).last_hidden_state
        return last_hidden_state

