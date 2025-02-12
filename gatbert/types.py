# STL
from typing import Dict, Literal
# 3rd Party
import torch

CorpusType = Literal['ezstance', 'semeval', 'vast', 'graph']

SampleType = Literal['token', 'graph', 'stripped_graph', 'concat']

AttentionType = Literal['edge_as_att', 'trans_key']

TensorDict = Dict[str, torch.Tensor]