# STL
from typing import List, Tuple, Dict
import abc
from collections import defaultdict, OrderedDict
import logging
from itertools import product
# 3rd party
import torch
from transformers import PreTrainedTokenizerFast
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


def build_identity_pool_mask(n: int):
    indices = [(i, i) for i in range(n)]
    values = [1 for _ in indices]
    return indices, values

def build_average_pool_mask(pool_inds: PoolIndices) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Based on the subword->node pooling indices,
    build a sparse mask matrix that will average the subwords' values
    for a given node.

    Args:
        pool_inds: Mapping from node IDs to subword IDs.
    Returns:
        A tuple containing the indices and values of a sparse matrix representing the mask.
        Indices are sorted.
    """
    mask_indices = []
    mask_values = []
    for (new_node_ind, subword_inds) in pool_inds.items():
        mask_indices.extend((new_node_ind, subword_ind) for subword_ind in subword_inds)
        v = 1 / len(subword_inds)
        mask_values.extend(v for _ in subword_inds)
    return mask_indices, mask_values

def encode_kb_nodes(tokenizer: PreTrainedTokenizerFast, kb: List[str], max_nodes = MAX_KB_NODES) -> Tuple[torch.Tensor, PoolIndices]:
    """
    Encodes knowledge-base nodes using tokenization. The pooling indices returned indicate which subword IDs contributed to which node
    """
    # FIXME: This is assuming all the KB tokens are conceptnet URIs
    clean_kb = [uri.split('/')[3] for uri in kb]
    clean_kb = [uri.replace("_", ' ') for uri in clean_kb]
    tokenized_kb = tokenizer(text=clean_kb,
                             is_split_into_words=True,
                             return_offsets_mapping=True,
                             return_special_tokens_mask=True,
                             return_tensors='pt')
    # Exclude the CLS and SEP tokens
    # FIXME: Works fine for BERT, might not work for Roberta and others
    real_inds = torch.where(~tokenized_kb['special_tokens_mask'].bool())
    tokenized_kb = {
        'input_ids': torch.unsqueeze(tokenized_kb['input_ids'][real_inds], dim=0),
        'offset_mapping': torch.unsqueeze(tokenized_kb['offset_mapping'][real_inds], dim=0)
    }
    # new_node_index -> [subword_indices]
    pool_inds = OrderedDict()

    new_nodes_index = -1
    subword_index = -1 # Has to be at least defined in case we skip the loop
    # For KB subwords, we plan to pool each into one combined node
    n_kb_nodes = 0
    for (subword_index, (start, end)) in enumerate(tokenized_kb['offset_mapping'].squeeze()):
        if start == 0:
            assert end != 0, "Special tokens should have been scrubbed"
            if n_kb_nodes >= max_nodes:
                __LOGGER.debug("Discarded %s/%s of external nodes", len(kb) - n_kb_nodes, len(kb))
                break
            n_kb_nodes += 1
            new_nodes_index += 1
            pool_inds[new_nodes_index] = []
        pool_inds[new_nodes_index].append(subword_index)
    else:
        # Needs to be 1 greater than the last subword we included
        subword_index += 1
    tokenized_kb['input_ids'] = tokenized_kb['input_ids'][..., :subword_index]
    return tokenized_kb['input_ids'], pool_inds


def collate_ids(tokenizer: PreTrainedTokenizerFast,
                samples: List[TensorDict],
                return_attention_mask: bool = False) -> TensorDict:
    token_padding = tokenizer.pad_token_id
    rdict = {}
    rdict['input_ids'] = torch.nn.utils.rnn.pad_sequence([torch.squeeze(s['input_ids'], 0) for s in samples],
                                                    batch_first=True, padding_value=token_padding)
    if return_attention_mask:
        rdict['attention_mask'] = rdict['input_ids'] != token_padding

    if 'position_ids' in samples[0]:
        # FIXME: Need a custom pad value for this?
        rdict['position_ids'] = torch.nn.utils.rnn.pad_sequence([torch.squeeze(s['position_ids'], 0) for s in samples],
                                                       batch_first=True)
    if 'token_type_ids' in samples[0]:
        rdict['token_type_ids'] = torch.nn.utils.rnn.pad_sequence([torch.squeeze(s['token_type_ids'], 0) for s in samples],
                                                       batch_first=True, padding_value=tokenizer.pad_token_type_id)
    return rdict

def collate_stance(samples: List[TensorDict]) -> TensorDict:
    return {'stance': torch.stack([s['stance'] for s in samples], dim=0)}

def collate_graph_data(samples: List[TensorDict],
                       return_node_counts: bool = False) -> TensorDict:
    max_subwords = -1

    node_counts = []
    new_edge_indices = []
    new_pool_indices = []
    new_pool_values = []
    for (i, s) in enumerate(samples):
        # edge indices
        edge_indices = s['edge_indices']
        edge_indices[0, :] = i
        new_edge_indices.append(edge_indices)

        # node mask
        pooling_mask = s['pooling_mask']
        (_, num_nodes, num_subwords) = pooling_mask.shape
        node_counts.append(num_nodes)
        max_subwords = max(max_subwords, num_subwords)
        indices = pooling_mask.indices()
        indices[0, :] = i
        new_pool_indices.append(indices)
        new_pool_values.append(pooling_mask.values())
    new_edge_indices = torch.concatenate(new_edge_indices, dim=-1)
    batch_node_mask = torch.sparse_coo_tensor(
        indices=torch.concatenate(new_pool_indices, dim=-1),
        values=torch.concatenate(new_pool_values, dim=-1),
        size=(len(samples), max(node_counts), max_subwords),
        device=pooling_mask.device,
        is_coalesced=True,
        requires_grad=True
    )
    rdict = {
        'pooling_mask': batch_node_mask,
        'edge_indices': new_edge_indices
    }
    if "node_type_ids" in rdict:
        rdict['node_type_ids'] = torch.nn.utils.rnn.pad_sequence([torch.squeeze(s['node_type_ids'], 0) for s in samples],
                                                       batch_first=True)
    if return_node_counts:
        rdict['node_counts'] = torch.tensor(node_counts, device=pooling_mask.device)
    return rdict




