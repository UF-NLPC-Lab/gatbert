# 3rd Party
import torch
from transformers import BertTokenizerFast, BertModel
# Local
from .encoder import SimpleEncoder
from .constants import DEFAULT_MODEL, Stance
from .base_module import StanceModule

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
        self.__encoder = SimpleEncoder(self.tokenizer)

    @property
    def encoder(self):
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


