# STL
from typing import Dict, Literal, Tuple
# 3rd Party
import torch

CorpusType = Literal['ezstance', 'semeval', 'vast']

TensorDict = Dict[str, torch.Tensor]

type AdjMat = Dict[int, Tuple[int, int]]