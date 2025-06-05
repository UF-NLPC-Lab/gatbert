# STL
from typing import List, Dict, Iterable, Tuple, Any
import abc
import logging
# 3rd party
import torch
from transformers import PreTrainedTokenizerFast
# Local
from .types import TensorDict
from .sample import Sample

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
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_context_length=256, max_target_length=64):
        self.__tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_target_length = max_target_length

    @property
    def tokenizer(self):
        return self.__tokenizer

    def encode(self, sample: Sample):
        rdict = encode_text(self.__tokenizer, sample, max_context_length=self.max_context_length, max_target_length=self.max_target_length)
        if sample.stance is not None:
            rdict['labels'] = torch.tensor([sample.stance.value])
        return rdict
    def collate(self, samples: List[TensorDict]) -> TensorDict:
        rdict= collate_ids(self.__tokenizer, samples, return_attention_mask=True)
        if 'labels' in samples[0]:
            rdict['labels'] = keyed_scalar_stack(samples, 'labels')
        return rdict


def keyed_pad(samples: List[TensorDict], k: str, padding_value=0):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.squeeze(s[k], dim=0) for s in samples],
        batch_first=True,
        padding_value=padding_value)

def keyed_scalar_stack(samples: List[TensorDict], k: str):
    return torch.stack([torch.squeeze(s[k]) for s in samples])

def encode_text(tokenizer: PreTrainedTokenizerFast,
                sample: Sample,
                max_context_length: int = 256, max_target_length: int = 64) -> TensorDict:
    tokenizer_kwargs = {'is_split_into_words': sample.is_split_into_words}
    context_trunc = tokenizer.decode(tokenizer.encode(sample.context, max_length=max_context_length, add_special_tokens=False), **tokenizer_kwargs)
    target_trunc = tokenizer.decode(tokenizer.encode(sample.target, max_length=max_target_length, add_special_tokens=False), **tokenizer_kwargs)
    combined = tokenizer(text=context_trunc, text_pair=target_trunc, return_tensors='pt', return_special_tokens_mask=True)
    return combined

def get_text_masks(special_tokens_mask):
    special_inds = torch.where(special_tokens_mask)[-1]
    seqlen = special_tokens_mask.shape[-1]
    all_inds = torch.arange(0, seqlen)
    ind_a = special_inds[0]
    ind_b = special_inds[1]
    if len(special_inds) > 3:
        # Handles BART--does <s>...</s></s>...</s> for a total of 4 special tokens
        ind_c = special_inds[-2]
        ind_d = special_inds[-1]
    elif len(special_inds) == 3:
        # Handles BERT-- [CLS]...[SEP]...[SEP]
        ind_c = ind_b
        ind_d = special_inds[-1]
    else:
        # Don't know what tokenizers would only give two special tokens
        ind_c = ind_b
        ind_d = seqlen
    context_text_mask = torch.logical_and(ind_a < all_inds, all_inds < ind_b)
    target_text_mask = torch.logical_and(ind_c < all_inds, all_inds < ind_d)
    return context_text_mask, target_text_mask

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
    if 'context_mask' in samples[0]:
        rdict['context_mask'] = keyed_pad(samples, 'context_mask', padding_value=False)
    if 'target_mask' in samples[0]:
        rdict['target_mask'] = keyed_pad(samples, 'target_mask', padding_value=False)
    return rdict

def collate_edge_indices(samples: Iterable[torch.Tensor]) -> torch.Tensor:
    batched = []
    for (i, s) in enumerate(samples):
        s[0, :] = i
        batched.append(s)
    return torch.concatenate(batched, dim=-1)