# STL
from typing import Dict, Literal
# 3rd Party
import torch

CorpusType = Literal['ezstance', 'semeval', 'vast', 'graph', 'graph_token']

TensorDict = Dict[str, torch.Tensor]