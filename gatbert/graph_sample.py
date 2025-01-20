# STL
from __future__ import annotations
import dataclasses
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, OrderedDict
from itertools import product
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast
# Local
from .constants import Stance, NodeType, TOKEN_TO_KB_RELATION_ID, TOKEN_TO_TOKEN_RELATION_ID


@dataclasses.dataclass
class Edge:
    # head_graph_index: int
    head_node_index: int
    # tail_graph_index: int
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

class GraphSample:

    class Builder:
        def __init__(self,
                     stance: Stance):

            self.stance = stance
            self.target: List[str] = []
            self.context: List[str] = []
            self.kb = ArrayDict()
            self.edges: Set[Edge] = set()

        def add_seeds(self,
                     target: List[Tuple[str, List[Any]]],
                     context: List[Tuple[str, List[Any]]]):
            """
            Add tokens and their matching seed concepts
            """

            self.target = [pair[0] for pair in target]
            self.context = [pair[0] for pair in context]

            node_lists = [pair[1] for pair in target + context]

            for (text_ind, nodes) in enumerate(node_lists):
                for node in nodes:
                    node_ind = self.kb.get_index(node)

                    self.edges.add(Edge(
                        head_graph_index=NodeType.TOKEN.value,
                        head_node_index=text_ind,
                        tail_graph_index=NodeType.KB.value,
                        tail_node_index=node_ind,
                        relation_id=TOKEN_TO_KB_RELATION_ID
                    ))
                    self.edges.add(Edge(
                        head_graph_index=NodeType.KB.value,
                        head_node_index=node_ind,
                        tail_graph_index=NodeType.TOKEN.value,
                        tail_node_index=text_ind,
                        relation_id=TOKEN_TO_KB_RELATION_ID
                    ))

        def add_kb_edge(self,
                        head: Any,
                        tail: Any,
                        relation_id: int):
            head_node_index = self.kb.get_index(head)
            tail_node_index = self.kb.get_index(tail)

            self.edges.add(Edge(
                head_graph_index=NodeType.KB.value,
                head_node_index=head_node_index,
                tail_graph_index=NodeType.KB.value,
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

    def encode(self, tokenizer: PreTrainedTokenizerFast) -> Dict[str, torch.Tensor]:

        # FIXME: This is assuming all the KB tokens are conceptnet URIs
        clean_kb = [uri.split('/')[3] for uri in self.kb]

        tokenized_text = tokenizer(text=self.target, text_pair=self.context, is_split_into_words=True, return_offsets_mapping=True, return_tensors='pt')
        tokenized_kb = tokenizer(text=clean_kb, is_split_into_words=True, return_offsets_mapping=True, return_tensors='pt')
        concat_ids = torch.concatenate([tokenized_text['input_ids'], tokenized_kb['input_ids']], dim=-1).squeeze()

        # (node_index, subword_index)
        expand_list = defaultdict(list)
        pool_inds = OrderedDict()

        new_nodes_index = -1
        orig_nodes_index = -1

        # For token subwords, we will split a token's nodes into subwords
        token_offset_mapping = tokenized_text['offset_mapping'].squeeze()
        # Handle splitting of token nodes into subword nodes
        for (subword_index, (start, end)) in enumerate(token_offset_mapping):
            new_nodes_index += 1
            pool_inds[new_nodes_index] = []

            if start != end: # Real character, not a special character
                if start == 0: # Start of a token
                    orig_nodes_index += 1
                expand_list[orig_nodes_index].append(new_nodes_index)
            pool_inds[new_nodes_index].append(subword_index)

        # For KB subwords, we plan to pool each into one combined node
        kb_offset_mapping = tokenized_kb['offset_mapping'].squeeze()
        for (subword_index, (start, end)) in enumerate(kb_offset_mapping, start=subword_index + 1):
            if start == end:
                # Special character; skip over
                new_nodes_index += 1
                pool_inds[new_nodes_index] = []
            elif start == 0:
                new_nodes_index += 1
                pool_inds[new_nodes_index] = []
                orig_nodes_index += 1
                expand_list[orig_nodes_index].append(new_nodes_index)
            pool_inds[new_nodes_index].append(subword_index)
        num_new_nodes = new_nodes_index + 1

        mask_indices = []
        mask_values = []
        for (new_node_ind, subword_inds) in pool_inds.items():
            mask_indices.extend((0, new_node_ind, subword_ind) for subword_ind in subword_inds)
            v = 1 / len(subword_inds)
            mask_values.extend(v for _ in subword_inds)

        mask_indices = torch.tensor(mask_indices).transpose(1, 0)
        mask_values = torch.tensor(mask_values)
        node_mask = torch.sparse_coo_tensor(
            indices=mask_indices,
            values=mask_values,
            size=(1, num_new_nodes, concat_ids.shape[-1]),
            is_coalesced=True,
            dtype=torch.float
        )

        # Indices into a sparse array (batch, max_new_nodes, max_new_nodes, relation)
        # Need a 0 at the beginning for batch
        new_edges = []
        # The original token-to-token edges of a standard BERT model
        num_text_tokens = tokenized_text.input_ids.shape[-1]
        new_edges.extend((0, head, tail, TOKEN_TO_TOKEN_RELATION_ID) for (head, tail) in product(range(num_text_tokens), range(num_text_tokens)))
        new_edges.extend((0, tail, head, TOKEN_TO_TOKEN_RELATION_ID) for (head, tail) in product(range(num_text_tokens), range(num_text_tokens)))

        # The edges that we read from the file.
        # Update their head/tail indices to account for subwords and special tokens
        for edge in self.edges:
            if edge.head_node_index not in expand_list:
                print(f"Warning: found no expansions for node {edge.head_node_index}")
                continue
            head_expand_list = expand_list[edge.head_node_index]
            if edge.tail_node_index not in expand_list:
                print(f"Warning: found no expansions for node {edge.tail_node_index}")
                continue
            tail_expand_list = expand_list[edge.tail_node_index]
            new_edges.extend((0, head, tail, edge.relation_id) for (head, tail) in product(head_expand_list, tail_expand_list))
        new_edges.sort()

        new_edges = torch.tensor(new_edges).transpose(1, 0)
        return {
            "input_ids" : concat_ids,
            "node_mask" : node_mask,
            "edge_indices": new_edges
        }

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
