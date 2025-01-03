# STL
from typing import List, Tuple
# 3rd Party
import torch
import numpy as np
# Local
from .constants import NUM_FAKE_NODES

EdgeList = List[Tuple[int, int, int]]
NodeList = List[int]

def make_fake_kb_links(n_text_nodes: int) -> Tuple[EdgeList, NodeList]:
    """
    
    Returns:
        Tuple of edges with 1 or more KB nodes, and a list of the KB nodes
    """
    relation_types = [
        0, # Token-to-Token
        1, # Token-to-CN
        2, # CN-to-Token
        3, # CN-to-CN
    ]
    node_universe = np.arange(1, NUM_FAKE_NODES + 1)
    edge_ids = []
    # Right now can only make fake KB nodes
    num_kb_nodes = torch.randint(3, 10, size=())
    chosen_nodes = torch.tensor(np.random.choice(node_universe, size=int(num_kb_nodes), replace=False))
    for (token_id, cn_id) in zip(*torch.where(torch.randn(n_text_nodes, num_kb_nodes) < .1)):
        edge_ids.append( (token_id, cn_id + n_text_nodes, 1) )
    for (token_id, cn_id) in zip(*torch.where(torch.randn(n_text_nodes, num_kb_nodes) < .1)):
        edge_ids.append( (cn_id + n_text_nodes, token_id, 2) )
    for (cn_id, cn_id_b) in zip(*torch.where(torch.randn(num_kb_nodes, num_kb_nodes) < .1)):
        edge_ids.append( (cn_id + n_text_nodes, cn_id_b + n_text_nodes, 3) )
    return edge_ids, chosen_nodes
