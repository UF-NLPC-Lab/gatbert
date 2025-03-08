# STL
from typing import Callable
# 3rd party
import torch

type CompOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class CgcnNodeUpdate(torch.nn.Module):

    def __init__(self,
                 comp_dim: int,
                 out_dim: int,
                 n_relations: int,
                 comp: CompOp = lambda h,r: h - r):
        super().__init__()
        # TODO: Update this to match the original paper and have
        # different projections for forward rels, inverse rels, and self-loops
        self.proj = torch.nn.Linear(comp_dim, out_dim, bias=False)
        self.comp = comp
        self.n_relations = n_relations
        self.out_dim = out_dim
        pass

    def forward(self, node_states, edge_indices, rel_states):
        (batch_size, num_nodes) = node_states.shape[:2]
        device = node_states.device

        # return node_states

        comps = self.comp(node_states[edge_indices[0], edge_indices[2]], rel_states[edge_indices[3]] )
        comps = self.proj(comps)

        # torch.sum on a sparse tensor doesn't support backpropagation, apparently
        # But coalescing over elements with identical indices does
        # So omit the indices for the tail node and the relation
        # That means we'll have sets of identical indices for head nodes,
        # whose values will get summed together when we coalesce
        summands = torch.sparse_coo_tensor(
            indices=edge_indices[:2],
            values=comps,
            size=(batch_size, num_nodes, self.out_dim),
            device=node_states.device,
            requires_grad=True,
            is_coalesced=False
        )
        summed = summands.coalesce().to_dense()

        # Similar principle applies for the counts
        uncoalesced_counts = torch.sparse_coo_tensor(
            indices=edge_indices[:2], # Only include batch and head node
            values=torch.ones(edge_indices.shape[-1], device=device),
            size=(batch_size, num_nodes),
            device=device,
            requires_grad=False,
            is_coalesced=False,
        )
        neighborhood_counts = uncoalesced_counts.coalesce().to_dense()

        average = summed / torch.unsqueeze(neighborhood_counts, -1)
        return average

class Cgcn(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_relations: int,
                 n_layers: int = 2,
                 comp: CompOp = lambda h,r: h - r):
        super().__init__()
        # TODO: Don't always assume that the relation will be a vector?
        self.graph_layers = torch.nn.ModuleList(
            [CgcnNodeUpdate(hidden_size, hidden_size, n_relations, comp) for _ in range(n_layers)]
        )
        self.relation_projects = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_layers)]
        )
    
    def forward(self,
                node_states,
                edge_indices,
                relation_states):
        for (graph_layer, rel_proj) in zip(self.graph_layers, self.relation_projects):
            node_states = graph_layer(node_states, edge_indices, relation_states)
            # FIXME: Need an activation function for the node_states
            relation_states = rel_proj(relation_states)
        return node_states, relation_states