# STL
from typing import Optional
# 3rd Party
import torch
from transformers import BartTokenizerFast
from transformers.models.bart.modeling_bart import BartEncoder, BartModel
# Local
from .encoder import SimpleEncoder
from .base_module import StanceModule
from .constants import Stance
from .bartenc_for_stance import BartEncForStance, BartEncForStanceConfig

class BartEncModule(StanceModule):

    def __init__(self,
                 pretrained_model: str = 'facebook/bart-large-mnli',
                 classifier_hidden_units: Optional[int] = None):
        super().__init__()
        config = BartEncForStanceConfig.from_pretrained(
            pretrained_model,
            classifier_hidden_units=classifier_hidden_units,
            id2label=Stance.id2label(),
            label2id=Stance.label2id(),
        )
        self.wrapped = BartEncForStance.from_pretrained(pretrained_model, config=config)
        self.__encoder = SimpleEncoder(BartTokenizerFast.from_pretrained(pretrained_model))

    @property
    def feature_size(self) -> int:
        return self.wrapped.config.d_model

    def get_optimizer_params(self):
        return [
            {"params": self.wrapped.encoder.parameters(), "lr": 2e-5},
            {"params": self.wrapped.classifier.parameters(), "lr": 1e-3}
        ]
    
    def forward(self, **kwargs):
        return self.wrapped(**kwargs)

    @property
    def encoder(self):
        return self.__encoder