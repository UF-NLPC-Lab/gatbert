from typing import List
# 3rd Party
import torch
from transformers import BertTokenizerFast, PreTrainedTokenizerFast, BertModel
# Local
from .sample import Sample, PretokenizedSample
from .encoder import encode_text, keyed_scalar_stack, collate_ids, Encoder, get_text_masks, keyed_pad
from .constants import DEFAULT_MODEL, Stance
from .base_module import StanceModule
from .types import TensorDict

class BertModule(StanceModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL,
                 dropout: float = 0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert.pooler = None
        hidden_size = self.bert.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(Stance), bias=True)
        )
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(pretrained_model)
        self.__encoder = self.Encoder(self.tokenizer)

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    @staticmethod
    def masked_average(mask, embeddings) -> torch.Tensor:
        # Need the epsilon to prevent divide-by-zero errors
        denom = torch.sum(mask, dim=-1, keepdim=True) + 1e-6
        return torch.sum(torch.unsqueeze(mask, -1) * embeddings, dim=-2) / denom

    def forward(self, **kwargs):
        # (1) Encode text
        bert_out = self.bert(**kwargs)
        feature_vec = bert_out.last_hidden_state[:, 0]
        logits = self.classifier(feature_vec)
        return logits

    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer
        def encode(self, sample: Sample | PretokenizedSample):
            return {
                **encode_text(self.__tokenizer, sample, 256, 256),
                'stance': torch.tensor([sample.stance.value])
            }
        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                **collate_ids(self.__tokenizer, samples, return_attention_mask=True),
                'stance': keyed_scalar_stack(samples, 'stance')
            }

