import torch
from .constants import SpecialRelation

IDENTITY_BASIS = 0

class RelationMatrixEmbeddings(torch.nn.Module):

    def __init__(self, n_relations: int, embedding_dim: int, num_bases: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        basis_data = torch.empty(num_bases, embedding_dim * embedding_dim)
        torch.nn.init.xavier_normal_(basis_data)
        # Make the first basis matrix an identity
        basis_data[IDENTITY_BASIS] = torch.flatten(torch.eye(embedding_dim))
        self.bases = torch.nn.parameter.Parameter(basis_data)

        self.coefficients = torch.nn.Embedding(n_relations, num_bases)
        coeff_data = self.coefficients.weight.data
        coeff_data[:, IDENTITY_BASIS] = 0
        coeff_data[SpecialRelation.TOKEN_TO_TOKEN.value, :] = 0
        coeff_data[SpecialRelation.TOKEN_TO_TOKEN.value, IDENTITY_BASIS] = 1


    def forward(self, edge_indices: torch.Tensor, batch_size: int, n_nodes: int):
        relation_ids = edge_indices[-1]

        # (num_edges, num_bases)
        basis_coeffs = self.coefficients(relation_ids)

        embedding_mat_vals = basis_coeffs @ self.bases
        embedding_mat_vals = embedding_mat_vals.reshape(*embedding_mat_vals.shape[:-1], self.embedding_dim, self.embedding_dim)

        edge_states = torch.sparse_coo_tensor(
            indices=edge_indices[:-1],
            values=embedding_mat_vals,
            size=(batch_size, n_nodes, n_nodes, *embedding_mat_vals.shape[-2:]),
            requires_grad=True,
            is_coalesced=False
        )
        edge_states = edge_states.coalesce()
        return edge_states
