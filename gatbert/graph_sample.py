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
                     target: OrderedDict[str, List[Any]],
                     context: OrderedDict[str, List[Any]]):
            """
            Add tokens and their matching seed concepts
            """

            self.target = [key for key in target]
            self.context = [key for key in context]
            n_text = len(self.target) + len(self.context)

            flatten_values = lambda d: [seed_list for seed_list in d.values()]
            node_lists = flatten_values(target) + flatten_values(context)

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
