import timm
from torch import nn
from dataclasses import dataclass


@dataclass
class TimmModelConfig:
    model_name: str = "timm/mobilenetv4_hybrid_medium.e500_r224_in1k"
    num_classes: int = 0
    hidden_dim: int = 1280

    def build_model(self):
        return TimmAdapter(
            self,
            timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=self.num_classes,
            ),
        )


class TimmAdapter(nn.Module):
    def __init__(self, config: TimmModelConfig, model=nn.Module):
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, x):
        return self.model(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def forward_head(self, x):
        return self.model.forward_head(x)
