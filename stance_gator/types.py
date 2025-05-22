# STL
from typing import Dict, Tuple
# 3rd Party
import torch

TensorDict = Dict[str, torch.Tensor]

type AdjMat = Dict[int, Tuple[int, int]]