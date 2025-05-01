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

class SimpleEncoder(Encoder):
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.__tokenizer = tokenizer
    def encode(self, sample: Sample | PretokenizedSample):
        return {
            **encode_text(self.__tokenizer, sample),
            'stance': torch.tensor([sample.stance.value])
        }
    def collate(self, samples: List[TensorDict]) -> TensorDict:
        return {
            **collate_ids(self.__tokenizer, samples, return_attention_mask=True),
            'stance': keyed_scalar_stack(samples, 'stance')
        }


def keyed_pad(samples: List[TensorDict], k: str, padding_value=0):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.squeeze(s[k], dim=0) for s in samples],
        batch_first=True,
        padding_value=padding_value)

def keyed_scalar_stack(samples: List[TensorDict], k: str):
    return torch.stack([torch.squeeze(s[k]) for s in samples])

def encode_text(tokenizer: PreTrainedTokenizerFast,
                sample: Sample | PretokenizedSample,
                max_context_length: int = 256, max_target_length: int = 256) -> TensorDict:
    if isinstance(sample, Sample):
        tokenizer_kwargs = {'is_split_into_words': False}
    elif isinstance(sample, GraphSample):
        tokenizer_kwargs = {'is_split_into_words': True}
        sample = sample.to_sample()
    elif isinstance(sample, PretokenizedSample):
        tokenizer_kwargs = {'is_split_into_words': True}
    else:
        raise ValueError(f"Invalid sample type {type(sample)}")
    context_trunc = tokenizer.decode(tokenizer.encode(sample.context, max_length=max_context_length, add_special_tokens=False), **tokenizer_kwargs)
    target_trunc = tokenizer.decode(tokenizer.encode(sample.target, max_length=max_target_length, add_special_tokens=False), **tokenizer_kwargs)
    combined = tokenizer(text=context_trunc, text_pair=target_trunc, return_tensors='pt', return_special_tokens_mask=True)
    return combined


def get_text_masks(special_tokens_mask):
    special_inds = torch.where(special_tokens_mask)[-1]
    seqlen = special_tokens_mask.shape[-1]
    cls_ind = special_inds[0]
    sep_ind = special_inds[1]
    if len(special_inds) > 2:
        end_ind = special_inds[2]
    else:
        end_ind = seqlen
    all_inds = torch.arange(0, seqlen)
    target_text_mask = torch.logical_and(cls_ind < all_inds, all_inds < sep_ind)
    context_text_mask = torch.logical_and(sep_ind < all_inds, all_inds < end_ind)
    return target_text_mask, context_text_mask

def collate_ids(tokenizer: PreTrainedTokenizerFast,
                samples: List[TensorDict],
                return_attention_mask: bool = False) -> TensorDict:
    token_padding = tokenizer.pad_token_id
    rdict = {}
    rdict['input_ids'] = keyed_pad(samples, 'input_ids', padding_value=token_padding)
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