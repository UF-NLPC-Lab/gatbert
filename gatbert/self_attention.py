import abc

import torch
from transformers.models.bert.modeling_bert import BertSelfAttention
# Local
from .config import GatbertConfig
from .constants import TOKEN_TO_TOKEN_RELATION_ID

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
                edge_indices: torch.Tensor):
        """
        Args:
            node_states: Strided tensor of shape (batch, nodes, hidden_state_size)
            edges: Hybrid array of shape (batch, nodes, nodes, hidden_state_size) where last dimension is dense
        """
        (batch_size, n_nodes) = node_states.shape[:2]

        logits = self.compute_attention_scores(node_states, edge_indices)
        sparse_attention = torch.sparse.softmax(logits, dim=-2)
        attention_vals = sparse_attention.values()
        attention_vals = self.dropout(attention_vals)

        values, V_edge_indices = self.compute_values(node_states, edge_indices)
        V_trans = self.transpose_for_scores(values) # (num_edges, num_edges, att_head_size)

        elwise_prod = torch.unsqueeze(attention_vals, -1) * V_trans # (num_edges, num_heads, att_head_size)
        elwise_prod = torch.flatten(elwise_prod, start_dim=1) # (num_edges, all_head_size)
        # (batch * nodes, num_edges)
        edge_mask = torch.sparse_coo_tensor(
            indices=torch.stack([V_edge_indices[0] * n_nodes + V_edge_indices[1], torch.arange(V_edge_indices.shape[1], device=V_edge_indices.device)]),
            values=torch.ones(V_edge_indices.shape[1]),
            size=(batch_size * n_nodes, V_edge_indices.shape[1]),
            device=elwise_prod.device,
            is_coalesced=True
        )
        # (batch * nodes, all_head_size)
        flat_node_states = torch.sparse.mm(edge_mask, elwise_prod)
        # (batch, nodes, all_head_size)
        node_states = flat_node_states.reshape(batch_size, n_nodes, -1)
        return node_states

    @abc.abstractmethod
    def load_pretrained_weights(self, other: BertSelfAttention):
        pass

    @abc.abstractmethod
    def compute_attention_scores(self, node_states, edge_indices):
        pass

    @abc.abstractmethod
    def compute_values(self, node_states, edge_indices):
        pass

class EdgeAsAttendeeSelfAttention(GatbertSelfAttention):
    """
    Self-attention mechanism inspired from https://arxiv.org/abs/2002.09685 with a few differences:
    - We have unique relation embeddings for every attention layer
    - They use the same key-projection matrix for their edges as their nodes; we do not

    The name was chosen because we are treating edges as keys and queries that can be attended over.

    Notice we don't explicitly project edges to any particular space; we instead just have direct embedding lookups.
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

    def compute_attention_scores(self, node_states, edge_indices):
        """
        Returns:
            sparse tensor of shape (batch_size, nodes, nodes, num_attention_heads)
        """
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
        return logits

    def compute_values(self, node_states, edge_indices):
        """

        Returns:
            tensor of shape (num_edges, hidden_size)
        """
        (batch_size, n_nodes) = node_states.shape[:2]
        V_node = self.value(node_states)
        V_edge = self.value_edge(edge_indices, batch_size, n_nodes)
        V_edge_indices = V_edge.indices() # FIXME: Why did we need to re-assign "edge_indices" ?
        V_sum = V_node[V_edge_indices[0], V_edge_indices[2]] + V_edge.values()
        return V_sum, V_edge_indices
