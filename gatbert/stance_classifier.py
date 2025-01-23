# 3rd Party
import torch
# Local
from .gatbert import GatbertModel
from .constants import Stance, NUMBER_RELATIONS

class StanceClassifier(torch.nn.Module):
    def __init__(self, config):
        self.bert = GatbertModel(config, NUMBER_RELATIONS)
        self.projection = torch.nn.Linear(
            config.hidden_size,
            out_features=len(Stance)
        )
    def forward(self, *args, **kwargs):
        (final_hidden_state, _) = self.bert(*args, **kwargs)
        logits = self.projection(final_hidden_state[:, 0])
        return logits