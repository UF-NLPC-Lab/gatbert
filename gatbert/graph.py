# STL
from typing import List, Tuple, Dict
import dataclasses
# 3rd Party
import torch
import numpy as np
# Local
from .constants import NUM_FAKE_NODES, NodeType, DummyRelationType

EdgeList = List[Tuple[int, int, int]]
NodeList = List[int]

@dataclasses.dataclass
class CNGraph:
    tok2id: Dict[str, int]
    id2uri: Dict[int, str]
    adj: Dict[int, List[Tuple[int, int]]]

    @staticmethod
    def from_json(json_data):
        return CNGraph(
            tok2id=json_data['tok2id'],
            id2uri={int(k):v for k,v in json_data['id2uri'].items()},
            adj   ={int(k):v for k,v in json_data['adj'].items()}
        )

def make_fake_kb_links(n_text_nodes: int) -> Tuple[EdgeList, NodeList]:
    """
    
    Returns:
        Tuple of edges with 1 or more KB nodes, and a list of the KB nodes
    """
    relation_types = [
        1, # Token-to-Token
        2, # Token-to-CN
        3, # CN-to-Token
        4, # CN-to-CN
    ]
    node_universe = np.arange(1, NUM_FAKE_NODES + 1)
    edge_ids = []
    # Right now can only make fake KB nodes
    num_kb_nodes = torch.randint(3, 10, size=())
    chosen_nodes = torch.tensor(np.random.choice(node_universe, size=int(num_kb_nodes), replace=False))
    for (token_id, cn_id) in zip(*map(lambda t: t.tolist(), torch.where(torch.randn(n_text_nodes, num_kb_nodes) < .1))):
        edge_ids.append( (token_id, cn_id, DummyRelationType.TOKEN_KB.value, NodeType.TOKEN.value, NodeType.KB.value) )
    for (token_id, cn_id) in zip(*map(lambda t: t.tolist(), torch.where(torch.randn(n_text_nodes, num_kb_nodes) < .1))):
        edge_ids.append( (cn_id, token_id, DummyRelationType.KB_TOKEN.value, NodeType.KB.value, NodeType.TOKEN.value) )
    for (cn_id, cn_id_b) in zip(*map(lambda t: t.tolist(), torch.where(torch.randn(num_kb_nodes, num_kb_nodes) < .1))):
        edge_ids.append( (cn_id, cn_id_b, DummyRelationType.KB_KB.value, NodeType.KB.value, NodeType.KB.value) )
    return edge_ids, chosen_nodes
