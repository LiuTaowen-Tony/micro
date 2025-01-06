import dataclasses
from typing import Union
import timm
import torchvision
import ml_utils.misc as misc
from torch import nn
from dataclasses import dataclass
from torch import LongTensor, FloatTensor
from transformers import AutoModel
from model.timm_adapter import TimmModelConfig
from model.transformer import DecoderOnlyTransformer, DecoderOnlyTransformerConfig
from model.transformers_adapter import TransformerAdapter, TransformerAdapterConfig
from sequence_modelling import boolean_triangular_mask


@dataclass
class ClipInitConfig:
    base_text_model: str = misc.relative_path(
        "../trained_models/lm/micro_model", __file__
    )
    image_model_config: TimmModelConfig = dataclasses.field(
        default_factory=lambda: TimmModelConfig(
            model_name="timm/mobilenetv4_hybrid_medium.e500_r224_in1k",
            num_classes=0,
            hidden_dim=1280,
        )
    )
    projection_dim: int = 256

    def build_model(self):
        if "micro" in self.base_text_model:
            text_model: DecoderOnlyTransformer = misc.load_with_config(
                DecoderOnlyTransformerConfig, self.base_text_model
            )
            hidden_dim = text_model.config.head_dim * text_model.config.num_heads
        else:
            text_model: TransformerAdapter = TransformerAdapterConfig(
                model_name=self.base_text_model, hidden_dim=768
            ).build_model()
            hidden_dim = text_model.config.hidden_dim

        clip_config = ClipConfig(
            text_model_config=text_model.config,
            image_model_config=self.image_model_config,
            projection_dim=self.projection_dim,
            image_hidden_dim=1280,
            text_hidden_dim=hidden_dim,
        )
        return Clip(
            clip_config,
            text_encoder=text_model,
            img_encoder=self.image_model_config.build_model(),
        )


@dataclass
class ClipConfig:
    text_model_config: Union[DecoderOnlyTransformerConfig, TransformerAdapterConfig]
    image_model_config: TimmModelConfig
    projection_dim: int = 256
    image_hidden_dim: int = 1280
    text_hidden_dim: int = 32000

    def build_model(self):
        return Clip(self)


class Clip(nn.Module):
    def __init__(self, config: ClipConfig, *, text_encoder, img_encoder):
        super().__init__()
        self.config = config
        self.text_encoder = text_encoder
        self.image_encoder = img_encoder 
        self.text_projection_head = ProjectionHead(
            self.config.text_hidden_dim, config.projection_dim
        )
        self.img_projection_head = ProjectionHead(
            self.config.image_hidden_dim, config.projection_dim
        )

    def forward(self, token_ids: LongTensor, image: FloatTensor, attention_mask):
        text_features = self.encode_text(token_ids, attention_mask)
        image_features = self.encode_image(image)
        return text_features, image_features

    def encode_text(self, token_ids: LongTensor, attention_mask):
        mask = boolean_triangular_mask(attention_mask)
        if isinstance(self.text_encoder, DecoderOnlyTransformer):
            embeddings = self.text_encoder.tok_embeddings(token_ids)
            logits = self.text_encoder.feature_transform(embeddings, mask=mask)
        elif isinstance(self.text_encoder, AutoModel):  # transformers model
            logits = self.text_encoder(token_ids, mask=mask)
        else:
            raise ValueError("Invalid text encoder")
        last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        text = self.text_projection_head(last_token_logits)
        return text / text.norm(dim=-1, keepdim=True)

    def encode_image(self, image: FloatTensor):
        encoded = self.image_encoder(image)
        img = self.img_projection_head(encoded)
        return img / img.norm(dim=-1, keepdim=True)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        h = self.projection(x)
        x = self.gelu(h)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + h
        x = self.layer_norm(x)
        return x
