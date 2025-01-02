# STL
from typing import Literal
# 3rd Party
import torch
# Local
from .basis_linear import BasisLinear

class RGATLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 attention_units: int,
                 out_features: int,
                 n_heads: int,
                 n_relations: int,
                 n_bases: int,
                 attention_mode: Literal['argat', 'wirgat'] = 'wirgat'):
        """
        Args:
            features:
                Number of input and output features
        """
        super().__init__()
        assert out_features % n_heads == 0
        assert attention_mode in {'argat', 'wirgat'}

        self.__in_features = in_features                          # F from RGAT
        self.__attention_units = attention_units                  # D from RGAT
        self.__chunked_features = out_features // n_heads         # F'/K 

        self.__n_bases = n_bases
        self.__n_relations = n_relations
        self.__n_heads = n_heads
        self.__attention_mode = attention_mode


        self.__projection = BasisLinear(
            in_features=(self.__in_features,),
            out_features=self.__chunked_features,
            n_bases=self.__n_bases,
            n_projections=[self.__n_relations, self.__n_heads]
        )

        # The query and key projections share the same bases
        self.__qk_proj = BasisLinear(
            in_features=(self.__n_relations, self.__n_heads, self.__chunked_features),
            out_features=self.__attention_units * 2,
            n_bases=self.__n_bases,
            n_projections=[self.__n_relations, self.__n_heads]
        )


    def forward(self, node_states, edges):
        """

        node_states: (batch, nodes, features)
        edges: sparse boolean array (batch, nodes, nodes, relations)
        """
        (batch_size, n_nodes) = node_states.shape[:-1]

        # (batch, nodes, relations, heads, chunked_features)
        V = self.__projection(node_states)

        # (batch, nodes, relations, heads 2, attention_units)
        QK = self.__qk_proj(V)
        # (batch, nodes, relations, heads, attention_units)
        Q, K = torch.split(QK, self.__attention_units, dim=-1)

        edge_indices = edges.indices()

        # (total_edges, num_heads, attention_units)
        Q_prime = Q[edge_indices[0], edge_indices[1], edge_indices[3], :, :]
        K_prime = K[edge_indices[0], edge_indices[2], edge_indices[3], :, :]
        # (total_edges, num_heads)
        logits = (Q_prime * K_prime).sum(dim=-1)

        if self.__attention_mode == 'wirgat':
            sparse_logits = torch.sparse_coo_tensor(
                indices=edge_indices, values=logits,
                size=(batch_size, n_nodes, n_nodes, self.__n_relations, self.__n_heads)
            )
            # Compute a separate probability distribution for each relation
            sparse_attention = torch.sparse.softmax(sparse_logits, dim=-3)
            # Validate probability distributions
            # assert torch.all(torch.abs(torch.sum(sparse_attention.to_dense(), -3) - 1) < 1e6)
        else:
            mask_indices = torch.stack([
                edge_indices[0],
                edge_indices[1],
                edge_indices[2] * self.__n_relations + edge_indices[3],
            ], dim=0)
            sparse_logits = torch.sparse_coo_tensor(
                indices=mask_indices, values=logits,
                size=(batch_size, n_nodes, n_nodes * self.__n_relations, self.__n_heads),
                requires_grad=True
            )
            sparse_attention = torch.sparse.softmax(sparse_logits, dim=-2)
            # Validate probability distributions
            # assert torch.all(torch.abs(torch.sum(sparse_attention.to_dense(), -2) - 1) < 1e6)
        edge_attentions = sparse_attention.values()
        

        # (batch, dest_nodes, relations, heads, chunk) --> (edges, heads, chunks)
        V_prime = V[edge_indices[0], edge_indices[2], edge_indices[3]]
        # (edges, heads, chunks)
        elwise_prod = torch.unsqueeze(edge_attentions, -1) * V_prime
        # (edges, out_features)
        elwise_prod = torch.flatten(elwise_prod, start_dim=1)

        # (batch_size*source_nodes, edges)
        edge_mask = torch.sparse_coo_tensor(
            indices=torch.stack([edge_indices[0]*n_nodes + edge_indices[1], torch.arange(edge_indices.shape[1])]),
            values=torch.ones(edge_indices.shape[1]),
            size=(batch_size * n_nodes, edge_indices.shape[1]),
            check_invariants=True
        )

        # (batch_size * source_nodes, out_features)
        flat_node_states = torch.sparse.mm(edge_mask, elwise_prod)

        # (batch_size, source_nodes, out_features)
        node_states = flat_node_states.view(batch_size, n_nodes, -1)
        return node_states
