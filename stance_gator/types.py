# STL
from typing import Dict, Literal, Tuple
# 3rd Party
import torch

CorpusType = Literal['ezstance', 'semeval', 'vast', 'graph']

AttentionType = Literal['edge_as_att', 'trans_key', 'rel_mat', 'hetero']

Transform = Literal['rm_external']

TensorDict = Dict[str, torch.Tensor]

type AdjMat = Dict[int, Tuple[int, int]]