# STL
from typing import Dict, Literal
# 3rd Party
import torch

CorpusType = Literal['ezstance', 'semeval', 'vast', 'graph']

AttentionType = Literal['edge_as_att', 'trans_key', 'rel_mat']

Transform = Literal['rm_external', 'cls_global_edges']

TensorDict = Dict[str, torch.Tensor]