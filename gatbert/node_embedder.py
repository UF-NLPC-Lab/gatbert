import abc
import torch

# TODO: Add support for position and token type embeddings

class NodeEmbedder(torch.nn.Module):

    @abc.abstractmethod
    def forward(self,
                subnode_ids: torch.Tensor,
                pooling_mask: torch.Tensor):
        raise NotImplementedError

class SimpleNodeEmbedder(NodeEmbedder):
    def __init__(self,
                 token_embedding: torch.nn.Embedding):
        self.__token_embedding = token_embedding

    def forward(self, subnode_ids: torch.Tensor, pooling_mask: torch.Tensor):
        if not pooling_mask.is_coalesced():
            pooling_mask = pooling_mask.coalesce()
        (batch_size, max_nodes, max_subnodes) = pooling_mask.shape

        # (batch_size, max_subnodes, embedding)
        subnode_embeddings = self.__token_embedding(subnode_ids)
        # (batch_size * max_subnodes, embedding)
        subnode_embeddings = torch.flatten(subnode_embeddings, end_dim=-2)


        indices = pooling_mask.get_indices()
        batch_node_indices = indices[0] * max_nodes + indices[1]
        batch_subnode_indices = indices[0] * max_subnodes + indices[2]
        # (batch_size * max_nodes, batch_size * max_subnodes)
        expanded_mask = torch.sparse_coo_tensor(
            indices=torch.stack([batch_node_indices, batch_subnode_indices]),
            values=pooling_mask.values(),
            size=(batch_size * max_nodes, batch_size * max_subnodes),
            is_coalesced=True,
            requires_grad=True,
            device=pooling_mask.device
        )

        # (batch_size * max_nodes, batch_size * max_subnodes) @ (batch_size * max_subnodes, embedding)
        node_embeddings = torch.sparse.mm(expanded_mask, subnode_embeddings)

        node_embeddings = node_embeddings.view(batch_size, max_nodes, -1)
        return node_embeddings
