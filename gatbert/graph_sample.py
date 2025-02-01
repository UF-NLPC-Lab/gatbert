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

    def encode(self, tokenizer: PreTrainedTokenizerFast) -> Dict[str, torch.Tensor]:
        assert tokenizer.pad_token_id == 0, "Only tokenizers that use 0-padding are supported"

        # FIXME: This is assuming all the KB tokens are conceptnet URIs
        clean_kb = [uri.split('/')[3] for uri in self.kb]
        clean_kb = [uri.replace("_", ' ') for uri in clean_kb]

        tokenized_text = tokenizer(text=self.target,
                                   text_pair=self.context,
                                   is_split_into_words=True,
                                   return_offsets_mapping=True,
                                   return_tensors='pt',
                                   truncation='longest_first')
        tokenized_kb = tokenizer(text=clean_kb,
                                 is_split_into_words=True,
                                 return_offsets_mapping=True,
                                 return_tensors='pt')
        device = tokenized_text['input_ids'].device
        position_ids = torch.tensor(
            [i for i in range(tokenized_text['input_ids'].shape[-1])] + \
            [0 for _ in range(tokenized_kb['input_ids'].shape[-1])],
            device=device
        )

        relevant_keys = ['input_ids', 'offset_mapping']
        tokenized_text = {k:tokenized_text[k] for k in relevant_keys}
        tokenized_kb = {k:tokenized_kb[k] for k in relevant_keys}
        # Assumes the tokens with 0-length offsets are special tokens
        real_toks = tokenized_kb['offset_mapping'][0, :, 1] != 0
        for k in relevant_keys:
            tokenized_kb[k] = tokenized_kb[k][:, real_toks]

        # old_node_index -> [new_node_indices]
        expand_list = defaultdict(list)
        # new_node_index -> [subword_indices]
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


        # Need to fast-forward past the token nodes to the external ones
        # Some of the token nodes may have been truncated by the tokenizer
        orig_nodes_index = len(self.target) + len(self.context) - 1

        # For KB subwords, we plan to pool each into one combined node
        kb_offset_mapping = tokenized_kb['offset_mapping'].squeeze()
        n_kb_nodes = 0
        for (subword_index, (start, end)) in enumerate(kb_offset_mapping, start=subword_index + 1):
            if start == 0:
                assert end != 0, "Special tokens should have been scrubbed"
                if n_kb_nodes >= MAX_KB_NODES:
                    GraphSample.LOGGER.debug("Discarded %s/%s of external nodes", len(self.kb) - n_kb_nodes, len(self.kb))
                    break
                n_kb_nodes += 1
                new_nodes_index += 1
                pool_inds[new_nodes_index] = []
                orig_nodes_index += 1
                expand_list[orig_nodes_index].append(new_nodes_index)
            pool_inds[new_nodes_index].append(subword_index)
        else:
            # Needs to be 1 greater than the last subword we included
            subword_index += 1

        concat_ids = torch.concatenate([tokenized_text['input_ids'], tokenized_kb['input_ids']], dim=-1).squeeze()
        # The tokenizer already did truncation for tokens, but this is where we do truncation for external nodes
        concat_ids = concat_ids[..., :subword_index]
        position_ids = position_ids[:subword_index]

        num_new_nodes = new_nodes_index + 1

        mask_indices = []
        mask_values = []
        for (new_node_ind, subword_inds) in pool_inds.items():
            mask_indices.extend((0, new_node_ind, subword_ind) for subword_ind in subword_inds)
            v = 1 / len(subword_inds)
            mask_values.extend(v for _ in subword_inds)

        mask_indices = torch.tensor(mask_indices, device=device).transpose(1, 0)
        mask_values = torch.tensor(mask_values, device=device)
        node_mask = torch.sparse_coo_tensor(
            indices=mask_indices,
            values=mask_values,
            size=(1, num_new_nodes, concat_ids.shape[-1]),
            is_coalesced=True,
            requires_grad=True,
            dtype=torch.float,
            device=device
        )

        # Indices into a sparse array (batch, max_new_nodes, max_new_nodes, relation)
        # Need a 0 at the beginning for batch
        new_edges = []
        # The original token-to-token edges of a standard BERT model
        num_text_tokens = tokenized_text['input_ids'].shape[-1]
        new_edges.extend((0, head, tail, TOKEN_TO_TOKEN_RELATION_ID) for (head, tail) in product(range(num_text_tokens), range(num_text_tokens)))

        # The edges that we read from the file.
        # Update their head/tail indices to account for subwords and special tokens
        discarded = 0
        for edge in self.edges:
            if edge.head_node_index not in expand_list or edge.tail_node_index not in expand_list:
                discarded += 1
                continue

            head_expansions = expand_list[edge.head_node_index]
            tail_expansions = expand_list[edge.tail_node_index]
            new_edges.extend((0, head, tail, edge.relation_id) for (head, tail) in product(head_expansions, tail_expansions))
        new_edges.sort()
        GraphSample.LOGGER.debug("Discarded %s/%s edges.", discarded, len(self.edges))

        new_edges = torch.tensor(new_edges, device=device).transpose(1, 0)
        return {
            "input_ids" : concat_ids,
            "position_ids": position_ids,
            "pooling_mask" : node_mask,
            "edge_indices": new_edges,
            "stance": torch.tensor(self.stance.value, device=device)
        }

    @staticmethod
    def collate(samples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        max_nodes = -1
        max_subwords = -1
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
            max_nodes    = max(max_nodes, num_nodes)
            max_subwords = max(max_subwords, num_subwords)
            indices = pooling_mask.indices()
            indices[0, :] = i
            new_pool_indices.append(indices)
            new_pool_values.append(pooling_mask.values())

        new_edge_indices = torch.concatenate(new_edge_indices, dim=-1)
        batch_node_mask = torch.sparse_coo_tensor(
            indices=torch.concatenate(new_pool_indices, dim=-1),
            values=torch.concatenate(new_pool_values, dim=-1),
            size=(len(samples), max_nodes, max_subwords),
            device=pooling_mask.device,
            is_coalesced=True,
            requires_grad=True
        )
        new_input_ids = torch.nn.utils.rnn.pad_sequence([s['input_ids'] for s in samples], batch_first=True)
        new_position_ids = torch.nn.utils.rnn.pad_sequence([s['position_ids'] for s in samples], batch_first=True)
        return {
            'input_ids': new_input_ids,
            'position_ids': new_position_ids,
            'pooling_mask': batch_node_mask,
            'edge_indices': new_edge_indices,
            "stance": torch.stack([s['stance'] for s in samples])
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
