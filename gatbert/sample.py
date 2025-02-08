from __future__ import annotations
import dataclasses
from typing import List
# 3rd Party
import torch
# Local
from .constants import Stance
from .types import TensorDict

def simple_collate(tokenizer, samples: List[TensorDict]) -> TensorDict:
    token_padding = tokenizer.pad_token_id
    type_padding = tokenizer.pad_token_type_id
    batched = {}
    batched['input_ids'] = torch.nn.utils.rnn.pad_sequence([s['input_ids'].squeeze() for s in samples], batch_first=True, padding_value=token_padding)
    if "token_type_ids" in samples[0]:
        batched['token_type_ids'] = torch.nn.utils.rnn.pad_sequence([s['token_type_ids'].squeeze() for s in samples], batch_first=True, padding_value=type_padding)
    batched['attention_mask'] = batched['input_ids'] != token_padding
    batched['stance'] = torch.stack([s['stance'] for s in samples], dim=0)
    return batched

@dataclasses.dataclass
class Sample:
    context: str
    target: str
    stance: Stance

    class Encoder:
        def __init__(self, tokenizer):
            self.__tokenizer = tokenizer
        def encode(self, sample: PretokenizedSample) -> TensorDict:
            result = self.__tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=True, return_tensors='pt')
            result['stance'] = torch.tensor(sample.stance.value, device=result['input_ids'].device)
            return result
        def collate(self, samples: TensorDict) -> TensorDict:
            return simple_collate(self.__tokenizer, samples)

@dataclasses.dataclass
class PretokenizedSample:
    context: List[str]
    target: List[str]
    stance: Stance

    class Encoder:
        def __init__(self, tokenizer):
            self.__tokenizer = tokenizer

        def encode(self, sample: PretokenizedSample) -> TensorDict:
            result = self.__tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=True, return_tensors='pt')
            result['stance'] = torch.tensor(sample.stance.value, device=result['input_ids'].device)
            return result
        
        def collate(self, samples: TensorDict) -> TensorDict:
            return simple_collate(self.__tokenizer, samples)