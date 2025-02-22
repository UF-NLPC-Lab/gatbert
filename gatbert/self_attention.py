import abc

import torch
from transformers.models.bert.modeling_bert import BertSelfAttention
# Local
from .config import GatbertConfig
from .constants import TOKEN_TO_TOKEN_RELATION_ID, NodeType
from .rel_mat_embeddings import RelationMatrixEmbeddings

def sparse_softmax_and_dropout(logits: torch.Tensor, dropout: torch.nn.Dropout):
    """
    Args:
        logits: sparse array of shape (batch, nodes, nodes, heads)
        dropout: torch dropout layer
    Returns:
        sparse array of probabilities of shape (batch, nodes, nodes, heads, 1)
    """
    sparse_attention = torch.sparse.softmax(logits, dim=-2)
    attention_vals = sparse_attention.values()
    attention_vals = dropout(attention_vals)
    return torch.sparse_coo_tensor(
        indices=logits.indices(),
        values=torch.unsqueeze(attention_vals, -1),
        size=(*logits.shape, 1),
        device=logits.device,
        is_coalesced=True,
        requires_grad=True
    )

class EdgeEmbeddings(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        super().__init__()
        self.embeddings = torch.nn.Embedding(config.n_relations, config.hidden_size)
        self.embeddings.weight.data[TOKEN_TO_TOKEN_RELATION_ID] = 0.

    def forward(self, edge_indices: torch.Tensor, batch_size: int, n_nodes: int):
        # All the indices save the relation IDs
        simple_edge_indices = edge_indices[:-1]
        relation_indices = edge_indices[-1]
        edge_values = self.embeddings(relation_indices)
        edge_states = torch.sparse_coo_tensor(
            indices=simple_edge_indices,
            values=edge_values,
            size=(batch_size, n_nodes, n_nodes, edge_values.shape[-1]),
            requires_grad=True,
            is_coalesced=False
        )
        # There will be duplicate values from multiple relations between the same nodes
        # Coalescing will sum these values
        edge_states = edge_states.coalesce()
        return edge_states

class GatbertSelfAttention(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_attention_heads: int = config.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0
        self.attention_head_size: int = self.hidden_size // self.num_attention_heads

    def compute_simple_values(self, node_states: torch.Tensor, projection: torch.nn.Linear) -> torch.Tensor:
        V = projection(node_states)
        V_trans = self.transpose_for_scores(V)
        return torch.unsqueeze(V_trans, dim=1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapted from BertSelfAttention HF class
        """
        # return BertSelfAttention.transpose_for_scores(self, x)
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x


    def forward(self,
                node_states: torch.Tensor,
                edge_indices: torch.Tensor,
                node_type_ids: torch.Tensor):
        """
        Args:
            node_states: Strided tensor of shape (batch, nodes, hidden_state_size)
            edge_indices: Array of shape (4, total_edges) where the 4 components are (batch_index, head_node_index, tail_node_index, relation_index)
        """
        attention_probs = self.compute_attention_probs(node_states, edge_indices, node_type_ids)
        values = self.compute_values(node_states, edge_indices, node_type_ids)

        # Why am I converting these to dense arrays?
        # torch.mul does indeed support sparse arrays, even if some dimensions have to be broadcast
        # Unfortunately, when doing backpropagation, the backward op will try to do a torch.unsqueeze or
        # similar on the broadcast operand, which is not yet supported by sparse matrices
        # If you're reading this years later, hopefully torch has better sparse support by now.
        attention_probs = attention_probs.to_dense()
        values = values.to_dense()
        prod = torch.mul(attention_probs, values)

        node_states = torch.sum(prod, dim=2) # Sum over tail nodes
        node_states = torch.flatten(node_states, start_dim=-2) # Concatenate across heads
        return node_states

    @abc.abstractmethod
    def load_pretrained_weights(self, other: BertSelfAttention):
        pass

    @abc.abstractmethod
    def compute_attention_probs(self, node_states, edge_indices, node_type_ids):
        """
        Computes attention probabilities (including any dropout) between head and tail nodes.

        Args:
            node_states: Strided tensor of shape (batch, nodes, hidden_state_size)
            edge_indices: Array of shape (4, total_edges) where the 4 components are (batch_index, head_node_index, tail_node_index, relation_index)
        
        Returns:
            Array of shape (batch, nodes, nodes, num_heads, 1). Intended to be element-wise multiplied with the result of compute_values.
        """

    @abc.abstractmethod
    def compute_values(self, node_states, edge_indices, node_type_ids):
        """
        Args:
            node_states: Strided tensor of shape (batch, nodes, hidden_state_size)
            edge_indices: Array of shape (4, total_edges) where the 4 components are (batch_index, head_node_index, tail_node_index, relation_index)

        Returns:
            An array of shape either (batch, 1, nodes, num_heads, att_head_size) or (batch, nodes, nodes, num_heads, att_head_size).
            In either case, dimension 1 represents the head node. 

            Intended to be element-wise multiplied with the result of compute_attention_scores.
        """

class HeterogeneousSelfAttention(GatbertSelfAttention):
    """
    Inspired from https://dl.acm.org/doi/abs/10.1145/3366423.3380027
    """
    def __init__(self, config, relation_embeddings: RelationMatrixEmbeddings):
        super().__init__(config)

        # Parameters already found in BERT models
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

        self.query_kb = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key_kb = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value_kb = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.relation_embeddings = relation_embeddings
        self.value_edge = EdgeEmbeddings(config)

        self.__sqrt_attention_size = torch.sqrt(torch.tensor(self.attention_head_size))
        self.__node_token_id = NodeType.TOKEN.value

    def load_pretrained_weights(self, other: BertSelfAttention):
        self.query.load_state_dict(other.query.state_dict())
        self.key.load_state_dict(other.key.state_dict())
        self.value.load_state_dict(other.value.state_dict())

    def __dual_project(self, tok_proj, kb_proj, states, node_type_ids):
        mask_a = torch.unsqueeze(node_type_ids == self.__node_token_id, -1)
        mask_b = torch.unsqueeze(node_type_ids != self.__node_token_id, -1)
        op_a = mask_a * tok_proj(states)
        op_b = mask_b * kb_proj(states)
        return op_a + op_b

    def compute_attention_probs(self, node_states, edge_indices, node_type_ids):
        """
        See section 3.2 of paper
        """
        (batch_size, n_nodes) = node_states.shape[:2]
        Q = self.__dual_project(self.query, self.query_kb, node_states, node_type_ids)
        K = self.__dual_project(self.key, self.key_kb, node_states, node_type_ids)

        edge_embeddings = self.relation_embeddings(edge_indices, batch_size, n_nodes)
        edge_indices = edge_embeddings.indices()
        edge_matrices = edge_embeddings.values()

        Q_trans = self.transpose_for_scores(Q[edge_indices[0], edge_indices[1]])
        K_trans = self.transpose_for_scores(K[edge_indices[0], edge_indices[2]])

        Q_trans_projected = Q_trans @ edge_matrices
        logit_vals = torch.einsum("ijk,ijk->ij", Q_trans_projected, K_trans) / self.__sqrt_attention_size
        logits = torch.sparse_coo_tensor(
            indices=edge_indices,
            values=logit_vals,
            size=(batch_size, n_nodes, n_nodes, self.num_attention_heads),
            is_coalesced=True,
            requires_grad=True
        )
        return sparse_softmax_and_dropout(logits, self.dropout)


    def compute_values(self, node_states, edge_indices, node_type_ids):
        """
        See Section 3.3 - Message Passing of Heterogeneous Graph Transformer paper
        """
        (batch_size, n_nodes) = node_states.shape[:2]
        V_node = self.__dual_project(self.value, self.value_kb, node_states, node_type_ids)
        V_edge = self.value_edge(edge_indices, batch_size, n_nodes)
        # Edge embeddings coalesces parallel edges with different relations, so we have fewer edges indices now
        V_edge_indices = V_edge.indices() 
        V_sum = V_node[V_edge_indices[0], V_edge_indices[2]] + V_edge.values()
        # (total_edges, num_heads, att_head_size)
        V_sum = self.transpose_for_scores(V_sum)
        values = torch.sparse_coo_tensor(
            indices=V_edge_indices,
            values=V_sum,
            size=(batch_size, n_nodes, n_nodes, self.num_attention_heads, self.attention_head_size),
            device=V_sum.device,
            is_coalesced=True,
            requires_grad=True
        )
        return values


class RelationInnerProdSelfAttention(GatbertSelfAttention):
    def __init__(self, config: GatbertConfig, relation_embeddings: RelationMatrixEmbeddings):
        super().__init__(config)
        self.relation_embeddings = relation_embeddings
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

        self.__sqrt_attention_size = torch.sqrt(torch.tensor(self.attention_head_size))

    def load_pretrained_weights(self, other: BertSelfAttention):
        assert self.num_attention_heads == other.num_attention_heads
        assert self.attention_head_size == other.attention_head_size
        assert self.hidden_size == other.query.in_features
        self.query.load_state_dict(other.query.state_dict())
        self.key.load_state_dict(other.key.state_dict())
        self.value.load_state_dict(other.value.state_dict())

    def compute_attention_probs(self, node_states, edge_indices, _):
        (batch_size, n_nodes) = node_states.shape[:2]
        Q = self.query(node_states)
        K = self.key(node_states)

        edge_embeddings = self.relation_embeddings(edge_indices, batch_size, n_nodes)
        edge_indices = edge_embeddings.indices()
        edge_matrices = edge_embeddings.values()

        Q_trans = self.transpose_for_scores(Q[edge_indices[0], edge_indices[1]])
        K_trans = self.transpose_for_scores(K[edge_indices[0], edge_indices[2]])

        Q_trans_projected = Q_trans @ edge_matrices
        logit_vals = torch.einsum("ijk,ijk->ij", Q_trans_projected, K_trans) / self.__sqrt_attention_size
        logits = torch.sparse_coo_tensor(
            indices=edge_indices,
            values=logit_vals,
            size=(batch_size, n_nodes, n_nodes, self.num_attention_heads),
            is_coalesced=True,
            requires_grad=True
        )
        return sparse_softmax_and_dropout(logits, self.dropout)


    def compute_values(self, node_states, _, _2):
        return self.compute_simple_values(node_states, self.value)

class TranslatedKeySelfAttention(GatbertSelfAttention):
    """
    Self-attention mechanism inspired from https://aclanthology.org/2022.coling-1.621/ with a few differences:
    - We don't have edge hidden states; rather, we look up new edge embeddings at each layer
    """

    def __init__(self, config):
        super().__init__(config)
        # Parameters already found in BERT models
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

        # New parameters
        self.key_edge = EdgeEmbeddings(config)

        self.__sqrt_attention_size = torch.sqrt(torch.tensor(self.attention_head_size))

    def load_pretrained_weights(self, other: BertSelfAttention):
        assert self.num_attention_heads == other.num_attention_heads
        assert self.attention_head_size == other.attention_head_size
        assert self.hidden_size == other.query.in_features
        self.query.load_state_dict(other.query.state_dict())
        self.key.load_state_dict(other.key.state_dict())
        self.value.load_state_dict(other.value.state_dict())

    def compute_attention_probs(self, node_states, edge_indices, _):
        (batch_size, n_nodes) = node_states.shape[:2]
        Q = self.query(node_states)
        K = self.key(node_states)

        edge_embeddings = self.key_edge(edge_indices, batch_size, n_nodes)
        edge_indices = edge_embeddings.indices() # Use these indices; we've coalesced across relations

        # Assuming a translational theory of embedding
        K_shifted = K[edge_indices[0], edge_indices[2]] - edge_embeddings.values()
        K_trans = self.transpose_for_scores(K_shifted) # (num_edges, head, att_head_size)

        Q_trans = self.transpose_for_scores(Q[edge_indices[0], edge_indices[1]]) # (num_edges, head, att_head_size)

        # Perform a dot-product over the att_head_size dimension
        # There's some overhead from calling einsum,
        # but there's also overhead from reshaping the tensors to call a regular inner product function
        logit_vals = torch.einsum("ijk,ijk->ij", Q_trans, K_trans) / self.__sqrt_attention_size


        logits = torch.sparse_coo_tensor(
            indices=edge_indices,
            values=logit_vals,
            size=(batch_size, n_nodes, n_nodes, self.num_attention_heads),
            device=logit_vals.device,
            is_coalesced=True,
            requires_grad=True
        )
        return sparse_softmax_and_dropout(logits, self.dropout)

    def compute_values(self, node_states, _, _2):
        return self.compute_simple_values(node_states, self.value)

class EdgeAsAttendeeSelfAttention(GatbertSelfAttention):
    """
    Self-attention mechanism inspired from https://arxiv.org/abs/2002.09685 with a few differences:
    - We have unique relation embeddings for every attention layer
    - They use the same key-projection matrix for their edges as their nodes; we do not
    - We do embedding lookups rather than "project" edges to a space (though a lookup basically is a projection)

    The name was chosen because we are treating edges as keys and queries that can be attended over.
    """

    def __init__(self, config):
        super().__init__(config)

        # Parameters already found in BERT models
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

        # New parameters
        self.key_edge = EdgeEmbeddings(config)
        self.value_edge = EdgeEmbeddings(config)

        self.__sqrt_attention_size = torch.sqrt(torch.tensor(self.attention_head_size))

    def load_pretrained_weights(self, other: BertSelfAttention):
        assert self.num_attention_heads == other.num_attention_heads
        assert self.attention_head_size == other.attention_head_size
        assert self.hidden_size == other.query.in_features
        self.query.load_state_dict(other.query.state_dict())
        self.key.load_state_dict(other.key.state_dict())
        self.value.load_state_dict(other.value.state_dict())

    def compute_attention_probs(self, node_states, edge_indices, _):
        Q = self.query(node_states)
        K_node = self.key(node_states)

        (batch_size, n_nodes) = node_states.shape[:2]
        K_edge = self.key_edge(edge_indices, batch_size, n_nodes)
        K_edge_indices = K_edge.indices() # FIXME: Why did we need to re-assign "edge_indices" ?
        K_edge_trans = self.transpose_for_scores(K_edge.values())

        Q_trans = self.transpose_for_scores(Q[K_edge_indices[0], K_edge_indices[1]])                       # (num_edges, num_heads, att_head_size)
        K_node_trans = self.transpose_for_scores(K_node[K_edge_indices[0], K_edge_indices[2]])             # (num_edges, num_heads, att_head_size)

        node2node = torch.sum(Q_trans * K_node_trans, dim=-1) / self.__sqrt_attention_size  # (num_edges, num_heads)
        node2edge = torch.sum(Q_trans * K_edge_trans, dim=-1) / self.__sqrt_attention_size  # (num_edges, num_heads)
        logits_val = node2node + node2edge
        # Hybrid tensor with two dense dimensions (num_attention_heads and attention head size)
        logits = torch.sparse_coo_tensor(
            indices=K_edge_indices,
            values=logits_val,
            size=(batch_size, n_nodes, n_nodes, self.num_attention_heads),
            device=logits_val.device,
            is_coalesced=True,
            requires_grad=True
        )
        sparse_attention = torch.sparse.softmax(logits, dim=-2)
        attention_vals = sparse_attention.values()
        attention_vals = self.dropout(attention_vals)
        dropout_att = torch.sparse_coo_tensor(
            indices=K_edge_indices,
            values=torch.unsqueeze(attention_vals, -1),
            size=(batch_size, n_nodes, n_nodes, self.num_attention_heads, 1),
            device=logits_val.device,
            is_coalesced=True,
            requires_grad=True
        )
        return dropout_att

    def compute_values(self, node_states, edge_indices, _):
        (batch_size, n_nodes) = node_states.shape[:2]
        V_node = self.value(node_states)
        V_edge = self.value_edge(edge_indices, batch_size, n_nodes)
        # Edge embeddings coalesces parallel edges with different relations, so we have fewer edges indices now
        V_edge_indices = V_edge.indices() 
        V_sum = V_node[V_edge_indices[0], V_edge_indices[2]] + V_edge.values()
        # (total_edges, num_heads, att_head_size)
        V_sum = self.transpose_for_scores(V_sum)
        values = torch.sparse_coo_tensor(
            indices=V_edge_indices,
            values=V_sum,
            size=(batch_size, n_nodes, n_nodes, self.num_attention_heads, self.attention_head_size),
            device=V_sum.device,
            is_coalesced=True,
            requires_grad=True
        )
        return values
