# STL
from __future__ import annotations
import dataclasses
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, OrderedDict
from itertools import product
import logging
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast
# Local
from .constants import Stance, TOKEN_TO_KB_RELATION_ID, TOKEN_TO_TOKEN_RELATION_ID, MAX_KB_NODES
from .sample import Sample, PretokenizedSample
from .types import TensorDict

PoolIndices = Dict[int, List[int]]

@dataclasses.dataclass
class Edge:
    head_node_index: int
    tail_node_index: int
    relation_id: int

    def __hash__(self):
        return hash(self.__to_tuple())
    def __eq__(self, other):
        return isinstance(other, Edge) and self.__to_tuple() == other.__to_tuple()

    def __to_tuple(self):
        return (self.head_node_index, self.tail_node_index, self.relation_id)

    def to_serial(self) -> str:
        return ','.join([str(el) for el in self.__to_tuple()])

    @staticmethod
    def from_serial(element: str):
        args = [int(e) for e in element.split(',')]
        return Edge(*args)

class ArrayDict:
    def __init__(self):
        self.data: str = []
        self.index: Dict[str, int] = {}
    
    def get_index(self, k):
        if k not in self.data:
            self.index[k] = len(self.data)
            self.data.append(k)
        return self.index[k]

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
                GraphSample.LOGGER.debug("Discarded %s/%s of external nodes", len(kb) - n_kb_nodes, len(kb))
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
    if return_node_counts:
        rdict['node_counts'] = torch.tensor(node_counts, device=pooling_mask.device)
    return rdict

class GraphSample:

    LOGGER = logging.getLogger("GraphSample")

    class Builder:
        """
        Builds the graph to be stored in the GraphSample.

        First call .add_seeds to add the text nodes and the external (i.e. ConceptNet)
        nodes they're directly linked to.
        The builder will make the necessary edges for these links.

        Then call .add_kb_edge for each edge that you have, in a breadth-first order.
        That is, the edges you pass to this method should be those obtained from a
        breadth-first search. This is crucial to tokenizing the the KB nodes later,
        as we will chop off the last nodes visited to stay within maximum sequence length.
        The closer nodes in a BFS are more important than the distant ones.
        """

        def __init__(self,
                     stance: Stance):

            self.stance = stance
            self.target: List[str] = []
            self.context: List[str] = []
            self.kb = ArrayDict()
            self.edges: Set[Edge] = set()

            self.__seeded = False

        def add_seeds(self,
                     target: List[Tuple[str, List[Any]]],
                     context: List[Tuple[str, List[Any]]]):
            """
            Add tokens and their matching seed concepts
            """

            self.target = [pair[0] for pair in target]
            self.context = [pair[0] for pair in context]
            n_text = len(self.target) + len(self.context)

            node_lists = [pair[1] for pair in target + context]

            for (text_ind, nodes) in enumerate(node_lists):
                for node in nodes:
                    node_ind = n_text + self.kb.get_index(node)

                    self.edges.add(Edge(
                        head_node_index=text_ind,
                        tail_node_index=node_ind,
                        relation_id=TOKEN_TO_KB_RELATION_ID
                    ))
                    self.edges.add(Edge(
                        head_node_index=node_ind,
                        tail_node_index=text_ind,
                        relation_id=TOKEN_TO_KB_RELATION_ID
                    ))
            self.__seeded = True

        def add_kb_edge(self,
                        head: Any,
                        tail: Any,
                        relation_id: int):
            if not self.__seeded:
                raise ValueError("Call .add_seeds before .add_kb_edge")

            n_text = len(self.target) + len(self.context)
            head_node_index = n_text + self.kb.get_index(head)
            tail_node_index = n_text + self.kb.get_index(tail)

            self.edges.add(Edge(
                head_node_index=head_node_index,
                tail_node_index=tail_node_index,
                relation_id=relation_id
            ))


        def build(self) -> GraphSample:
            return GraphSample(stance=self.stance,
                               target=self.target,
                               context=self.context,
                               kb=self.kb.data,
                               edges=list(self.edges)
            )

    class ConcatEncoder:
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer
            self.__text_encoder = PretokenizedSample.Encoder(tokenizer)

        def encode(self, sample: GraphSample):

            input_ids, pool_inds = encode_kb_nodes(self.__tokenizer, sample.kb)
            mask_indices, mask_values = build_average_pool_mask(pool_inds)

            mask_indices = [(0, *index) for index in mask_indices] # Prepend batch dim

            device = input_ids.device
            node_mask = torch.sparse_coo_tensor(
                indices=torch.tensor(mask_indices, device=device).transpose(1, 0),
                values=torch.tensor(mask_values, device=device),
                size=(1, len(sample.kb), input_ids.shape[-1]),
                is_coalesced=True,
                requires_grad=True,
                dtype=torch.float,
                device=device
            )

            orig_text_nodes = len(sample.target) + len(sample.context)
            # Only keep edges between two graph concepts
            iter_edge = filter(lambda e: e.head_node_index >= orig_text_nodes and e.tail_node_index >= orig_text_nodes, sample.edges)
            iter_edge = map(lambda e: (0, e.head_node_index - orig_text_nodes, e.tail_node_index - orig_text_nodes, e.relation_id), iter_edge) 
            edge_indices = sorted(iter_edge)
            if edge_indices:
                edge_indices = torch.tensor(edge_indices, device=device).transpose(1, 0)
            else:
                edge_indices = torch.empty([4, 0], dtype=torch.int, device=device)


            encoded_text = self.__text_encoder.encode(sample.to_sample())
            stance = encoded_text.pop('stance')
            return {
                "text": encoded_text,
                "graph": {
                    "input_ids" : input_ids,
                    "pooling_mask" : node_mask,
                    "edge_indices": edge_indices
                },
                'stance': stance
            }

        def collate(self, samples: List[Dict[str, TensorDict]]) -> TensorDict:
            rdict = {}
            rdict["stance"] = torch.stack([s['stance'] for s in samples])

            graph_samples = [s['graph'] for s in samples]
            rdict['graph'] = {
                **collate_ids(self.__tokenizer, graph_samples),
                **collate_graph_data(graph_samples, return_node_counts=True)
            }
            rdict['text'] = collate_ids(self.__tokenizer, [s['text'] for s in samples], return_attention_mask=True)
            return rdict

    class Encoder:
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer

        def encode(self, sample: GraphSample):
            tokenizer = self.__tokenizer

            tokenized_text = tokenizer(text=sample.target,
                                       text_pair=sample.context,
                                       is_split_into_words=True,
                                       return_offsets_mapping=True,
                                       return_tensors='pt',
                                       truncation='longest_first')
            device = tokenized_text['input_ids'].device

            kb_input_ids, kb_pool_inds = encode_kb_nodes(tokenizer, sample.kb)
            num_kb_nodes = max(kb_pool_inds) + 1 if kb_pool_inds else 0

            num_text_nodes = tokenized_text['input_ids'].shape[-1]
            num_text_subwords = num_text_nodes

            # Combine input ids
            concat_ids = torch.concatenate([tokenized_text['input_ids'], kb_input_ids], dim=-1)
            # Add dummy position ids for graph nodes
            position_ids = torch.tensor(
                [i for i in range(tokenized_text['input_ids'].shape[-1])] + \
                [0 for _ in range(kb_input_ids.shape[-1])],
                device=device
            )
            # Add dummy token_type ids for graph nodes
            last_token_type_id = tokenized_text['token_type_ids'][..., -1][0]
            token_type_ids = torch.concatenate([
                tokenized_text['token_type_ids'],
                torch.full_like(kb_input_ids, last_token_type_id)
                ],
                dim=-1
            )

            text_mask_inds, text_mask_values = build_identity_pool_mask(num_text_nodes)
            kb_mask_inds, kb_mask_values = build_average_pool_mask(kb_pool_inds)
            kb_mask_inds = [(i + num_text_nodes, j + num_text_subwords) for (i, j) in kb_mask_inds]

            mask_inds = text_mask_inds + kb_mask_inds
            mask_inds = [(0, *inds) for inds in mask_inds]
            
            mask_inds = torch.tensor(mask_inds, device=device).transpose(1, 0)
            mask_vals = torch.tensor(text_mask_values + kb_mask_values, device=device)
            total_pooled_nodes = num_text_nodes + num_kb_nodes
            node_mask = torch.sparse_coo_tensor(
                indices=mask_inds,
                values=mask_vals,
                size=(1, total_pooled_nodes, concat_ids.shape[-1]),
                is_coalesced=True,
                requires_grad=True,
                dtype=torch.float,
                device=device
            )

            # old_node_index -> [new_node_indices]
            expand_list = defaultdict(list)
            # For token subwords, we will split a token's nodes into subwords
            # Handle splitting of token nodes into subword nodes
            orig_nodes_index = -1
            for (new_nodes_index, (start, end)) in enumerate(tokenized_text['offset_mapping'].squeeze()):
                if start != end: # Real character, not a special character
                    if start == 0: # Start of a token
                        orig_nodes_index += 1
                    expand_list[orig_nodes_index].append(new_nodes_index)


            # Indices into a sparse array (batch, max_new_nodes, max_new_nodes, relation)
            # Need a 0 at the beginning for batch
            new_edges = []
            # The original token-to-token edges of a standard BERT model
            new_edges.extend((0, head, tail, TOKEN_TO_TOKEN_RELATION_ID) for (head, tail) in product(range(num_text_subwords), range(num_text_subwords)))
            # The KB edges, with indices adjusted
            orig_text_nodes = len(sample.target) + len(sample.context)
            max_node_index = orig_text_nodes + num_kb_nodes
            for e in sample.edges:
                if orig_text_nodes <= e.head_node_index < max_node_index:
                    head_list = [num_text_subwords + (e.head_node_index - orig_text_nodes)]
                elif e.head_node_index in expand_list:
                    head_list = expand_list[e.head_node_index]
                else:
                    continue
                if orig_text_nodes <= e.tail_node_index < max_node_index:
                    tail_list = [num_text_subwords + (e.tail_node_index - orig_text_nodes)]
                elif e.tail_node_index in expand_list:
                    tail_list = expand_list[e.tail_node_index]
                else:
                    continue
                new_edges.extend((0, head, tail, e.relation_id) for (head, tail) in product(head_list, tail_list))
            new_edges.sort()
            new_edges = torch.tensor(new_edges, device=device).transpose(1, 0)

            return {
                "input_ids" : concat_ids,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "pooling_mask" : node_mask,
                "edge_indices": new_edges,
                "stance": torch.tensor(sample.stance.value, device=device)
            }

        def collate(self, samples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {
                "stance": torch.stack([s['stance'] for s in samples]),
                **collate_ids(self.__tokenizer, samples),
                **collate_graph_data(samples)
            }

    def __init__(self,
                 stance: Stance,
                 target: List[str],
                 context: List[str],
                 kb: List[str],
                 edges: List[Edge]):
        self.stance = stance
        self.target = target
        self.context = context
        self.kb = kb
        self.edges = edges

    @staticmethod
    def from_pretokenized(s: PretokenizedSample):
        return GraphSample(stance=s.stance,
                           target=s.target,
                           context=s.context,
                           kb=[],
                           edges=[])

    def strip_external(self) -> GraphSample:
        return GraphSample(
            stance=self.stance,
            target=self.target,
            context=self.context,
            kb=[],
            edges=[]
        )

    def to_sample(self) -> PretokenizedSample:
        return PretokenizedSample(context=self.context, target=self.target, stance=self.stance)

    def to_row(self) -> List[str]:
        rval= [str(self.stance.value),
                str(len(self.target)),
                str(len(self.context)),
                str(len(self.kb))] \
            + self.target \
            + self.context \
            + self.kb \
            + [e.to_serial() for e in self.edges]
        return rval

    @staticmethod
    def from_row(entries: List[str]) -> GraphSample:
        stance = Stance(int(entries[0]))
        n_target = int(entries[1])
        n_context = int(entries[2])
        n_external = int(entries[3])

        target_end = 4 + n_target
        context_end = target_end + n_context
        nodes_end = context_end + n_external

        target = entries[4:target_end]
        context = entries[target_end:context_end]
        external_nodes = entries[context_end:nodes_end]
        edges = [Edge.from_serial(el) for el in entries[nodes_end:]]

        return GraphSample(
            target=target,
            context=context,
            kb=external_nodes,
            edges=edges,
            stance=stance
        )
