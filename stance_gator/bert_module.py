# STL
from typing import Optional
# 3rd Party
import torch
from transformers import BertTokenizerFast, BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig
# Local
from .bert_for_stance import BertForStance, BertForStanceConfig
from .encoder import SimpleEncoder
from .constants import DEFAULT_MODEL, Stance
from .base_module import StanceModule

class BertModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 classifier_hidden_units: Optional[int] = None
                 ):
        super().__init__()
        config = BertForStanceConfig.from_pretrained(pretrained_model,
                                                     classifier_hidden_units=classifier_hidden_units,
                                                     id2label=Stance.id2label(),
                                                     label2id=Stance.label2id(),
                                                     )
        self.wrapped = BertForStance.from_pretrained(pretrained_model, config=config)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = SimpleEncoder(self.tokenizer)
    @property
    def encoder(self):
        return self.__encoder
    @property
    def feature_size(self) -> int:
        return self.wrapped.config.hidden_size
    def forward(self, **kwargs):
        return self.wrapped(**kwargs)
        

