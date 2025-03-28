# STL
from typing import List, Tuple, Dict, Iterable
import abc
import logging
# 3rd party
import torch
from transformers import PreTrainedTokenizerFast, BertTokenizerFast
# Local
from .types import TensorDict
from .sample import Sample, PretokenizedSample
from .graph_sample import GraphSample

PoolIndices = Dict[int, List[int]]

__LOGGER = logging.getLogger("Encoder")

class Encoder(abc.ABC):

    @abc.abstractmethod
    def encode(self, sample) -> TensorDict:
        pass

    @abc.abstractmethod
    def collate(self, samples: List[TensorDict]) -> TensorDict:
        pass

def keyed_pad(samples: List[TensorDict], k: str, padding_value=0):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.squeeze(s[k], dim=0) for s in samples],
        batch_first=True,
        padding_value=padding_value)

def keyed_scalar_stack(samples: List[TensorDict], k: str):
    return torch.stack([torch.squeeze(s[k]) for s in samples])

def encode_text(tokenizer: PreTrainedTokenizerFast,
                sample: Sample | PretokenizedSample,
                tokenizer_kwargs = dict()) -> TensorDict:
    if isinstance(sample, Sample):
        return tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=False, return_tensors='pt', **tokenizer_kwargs)
    elif isinstance(sample, GraphSample):
        sample = sample.to_sample()
        return tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=True, return_tensors='pt', **tokenizer_kwargs)
    elif isinstance(sample, PretokenizedSample):
        return tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=True, return_tensors='pt', **tokenizer_kwargs)
    else:
        raise ValueError(f"Invalid sample type {type(sample)}")

def collate_ids(tokenizer: PreTrainedTokenizerFast,
                samples: List[TensorDict],
                return_attention_mask: bool = False) -> TensorDict:
    token_padding = tokenizer.pad_token_id
    rdict = {}
    rdict['input_ids'] = keyed_pad(samples, 'input_ids')
    if return_attention_mask:
        rdict['attention_mask'] = rdict['input_ids'] != token_padding
    if 'position_ids' in samples[0]:
        # FIXME: Need a custom pad value for this?
        rdict['position_ids'] = keyed_pad(samples, 'position_ids')
    if 'token_type_ids' in samples[0]:
        rdict['token_type_ids'] = keyed_pad(samples, 'token_type_ids', padding_value=tokenizer.pad_token_type_id)
    return rdict

def collate_edge_indices(samples: Iterable[torch.Tensor]) -> torch.Tensor:
    batched = []
    for (i, s) in enumerate(samples):
        s[0, :] = i
        batched.append(s)
    return torch.concatenate(batched, dim=-1)

def pretokenize_cn_uri(uri: str) -> List[str]:
    if uri.startswith('/'):
        return uri.split('/')[3].split('_')
    return uri.split()