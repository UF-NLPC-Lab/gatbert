# STL

# 3rd Party
import torch
from transformers import BartTokenizerFast
from transformers.models.bart.modeling_bart import BartEncoder, BartModel
# Local
from .encoder import SimpleEncoder
from .base_module import StanceModule
from .constants import Stance

class BartEncModule(StanceModule):

    def __init__(self,
                 pretrained_model: str = 'facebook/bart-large-mnli',
                 dropout: float = 0.2):
        super().__init__()
        self.bart: BartEncoder = BartModel.from_pretrained(pretrained_model).encoder
        self.bart.pooler = None

        hidden_size = self.bart.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(Stance), bias=True)
        )
        self.__encoder = SimpleEncoder(BartTokenizerFast.from_pretrained(pretrained_model))

    def configure_optimizers(self):
        # Values drawn from the EZ-stance paper
        grouped_params = [
            {"params": list(self.bart.parameters()), "lr": 2e-5},
            {"params": list(self.classifier.parameters()), "lr": 1e-3}
        ]
        return torch.optim.Adam(grouped_params)
    
    def forward(self, **kwargs):
        bart_out = self.bart(**kwargs)
        feature_vec = bart_out.last_hidden_state[:, 0]
        logits = self.classifier(feature_vec)
        return logits

    @property
    def encoder(self):
        return self.__encoder