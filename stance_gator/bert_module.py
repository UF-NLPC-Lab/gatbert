# STL
from typing import Optional
# 3rd Party
import torch
from transformers import BertTokenizerFast
# Local
from .models import BertForStance, BertForStanceConfig
from .encoder import SimpleEncoder
from .constants import DEFAULT_MODEL
from .base_module import StanceModule

class BertModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 classifier_hidden_units: Optional[int] = None,
                 max_context_length=256,
                 max_target_length=64,
                 **parent_kwargs
                 ):
        super().__init__(**parent_kwargs)
        config = BertForStanceConfig.from_pretrained(pretrained_model,
                                                     classifier_hidden_units=classifier_hidden_units,
                                                     id2label=self.stance_enum.id2label(),
                                                     label2id=self.stance_enum.label2id(),
                                                     )
        self.wrapped = BertForStance.from_pretrained(pretrained_model, config=config)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = SimpleEncoder(self.tokenizer, max_context_length=max_context_length, max_target_length=max_target_length, use_weights=True)
    @property
    def encoder(self):
        return self.__encoder
    @property
    def feature_size(self) -> int:
        return self.wrapped.config.hidden_size
    def forward(self, **kwargs):
        labels = kwargs.pop('labels', None)
        weights = kwargs.pop('weights', None)
        bert_output: BertForStance.Output = self.wrapped(**kwargs)

        if labels is not None:
            if weights is None:
                weights = torch.ones_like(labels, dtype=torch.float)
            loss_values = torch.nn.functional.cross_entropy(
                bert_output.logits,
                target=labels,
                reduction='none'
            )
            scaled_losses = weights * loss_values
            loss = torch.mean(scaled_losses)
            bert_output.loss = loss
        return bert_output
        

