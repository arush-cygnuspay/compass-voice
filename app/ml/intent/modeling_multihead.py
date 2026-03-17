# app/ml/intent/modeling_multihead.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig


class MultiHeadConfig(PretrainedConfig):
    model_type = "multihead_intent"

    def __init__(
        self,
        base_model_name_or_path: str = "distilbert-base-uncased",
        num_main_labels: int = 6,
        num_sub_labels: int = 37,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.num_main_labels = num_main_labels
        self.num_sub_labels = num_sub_labels
        self.dropout = dropout


class MultiHeadIntentModel(PreTrainedModel):
    config_class = MultiHeadConfig

    def __init__(self, config: MultiHeadConfig):
        super().__init__(config)

        base_cfg = AutoConfig.from_pretrained(config.base_model_name_or_path)
        self.encoder = AutoModel.from_config(base_cfg)

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifier_main = nn.Linear(hidden_size, config.num_main_labels)
        self.classifier_sub = nn.Linear(hidden_size, config.num_sub_labels)

        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def init_encoder_from_pretrained(self) -> None:
        self.encoder = AutoModel.from_pretrained(self.config.base_model_name_or_path)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels_main: Optional[torch.Tensor] = None,
        labels_sub: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT-style CLS token representation
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        logits_main = self.classifier_main(pooled)
        logits_sub = self.classifier_sub(pooled)

        loss = None
        if labels_main is not None and labels_sub is not None:
            loss_main = self.loss_fn(logits_main, labels_main)
            loss_sub = self.loss_fn(logits_sub, labels_sub)
            loss = loss_main + loss_sub

        return {
            "loss": loss,
            "logits_main": logits_main,
            "logits_sub": logits_sub,
        }