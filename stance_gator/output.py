# STL
import dataclasses
from typing import Optional
# 3rd party
import torch
from transformers.utils.generic import ModelOutput

@dataclasses.dataclass
class StanceOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seq_encoding: Optional[torch.FloatTensor] = None
