# 3rd Party
import torch
from transformers import AutoModel, BertModel
# Local
from .gatbert import GatbertModel
from .constants import Stance

class GraphClassifier(torch.nn.Module):
    def __init__(self, config, n_relations):
        super().__init__()
        self.bert = GatbertModel(config, n_relations=n_relations)
        self.projection = torch.nn.Linear(
            config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
    def forward(self, input_ids: torch.Tensor, pooling_mask: torch.Tensor, edge_indices: torch.Tensor):
        final_hidden_state = self.bert(input_ids, pooling_mask, edge_indices)
        logits = self.projection(final_hidden_state[:, 0])
        return logits

class BertClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.projection = torch.nn.Linear(
            self.bert.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
    def forward(self, *args, **kwargs):
        bert_output = self.bert(*args, **kwargs)
        last_hidden_state = bert_output['last_hidden_state'][:, 0]
        logits = self.projection(last_hidden_state)
        return logits