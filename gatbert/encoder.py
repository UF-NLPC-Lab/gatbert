# STL
from typing import List, Tuple, Dict, Iterable
import abc
from collections import defaultdict, OrderedDict
import logging
from itertools import product
# 3rd party
import torch
from transformers import PreTrainedTokenizerFast, BertTokenizerFast
# Local
from .types import TensorDict
from .constants import MAX_KB_NODES, TOKEN_TO_TOKEN_RELATION_ID, TOKEN_TO_KB_RELATION_ID
from .data import get_default_pretokenize
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

def extract_kb_edges(sample: GraphSample) -> torch.Tensor:
    orig_text_nodes = len(sample.target) + len(sample.context)
    # Only keep edges between two graph concepts
    iter_edge = filter(lambda e: e.head_node_index >= orig_text_nodes and e.tail_node_index >= orig_text_nodes, sample.edges)
    iter_edge = map(lambda e: (0, e.head_node_index - orig_text_nodes, e.tail_node_index - orig_text_nodes, e.relation_id), iter_edge) 
    edge_indices = sorted(iter_edge)
    if edge_indices:
        edge_indices = torch.tensor(edge_indices).transpose(1, 0)
    else:
        edge_indices = torch.empty([4, 0], dtype=torch.int)
    return edge_indices

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
