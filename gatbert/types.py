# STL
from typing import Dict, Literal
# 3rd Party
import torch

CorpusType = Literal['ezstance', 'semeval', 'vast', 'graph']

TensorDict = Dict[str, torch.Tensor]