
# 3rd Party
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention, \
    BertSelfOutput, \
    BertIntermediate, \
    BertOutput, \
    BertAttention, \
    BertLayer, \
    BertEncoder, \
    BertEmbeddings, \
    BertModel

class GatbertSelfAttention(torch.nn.Module):
    """
    Attention calculation among nodes, modulated by a sparse set of edge features
    """

    def __init__(self, config):
        self.hidden_size: int = config.hidden_size
        self.num_attention_heads: int = config.num_attention_heads
        self.attention_head_size: int = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob: float = self.dropout_prob

        # Parameters already found in BERT models
        self.query = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = torch.nn.Dropout(self.dropout_prob)

        # New parameters
        self.value_edge = torch.nn.Linear(self.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapted from BertSelfAttention HF class
        """
        new_shape = x.shape[:1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1)

    def load_pretrained_weights(self, other: BertSelfAttention):
        assert self.num_attention_heads == other.num_attention_heads
        assert self.attention_head_size == other.attention_head_size
        assert self.hidden_size == other.query.in_features
        self.query.load_state_dict(other.query.state_dict())
        self.key.load_state_dict(other.key.state_dict())
        self.value.load_state_dict(other.value.state_dict())

    def forward(self,
                node_states: torch.Tensor,
                edge_states: torch.Tensor):
        """
        Args:
            node_states: Strided tensor of shape (batch, nodes, hidden_state_size)
            edges: Hybrid array of shape (batch, nodes, nodes, hidden_state_size) where last dimension is dense
        """

        Q = self.query(node_states)
        K_node = self.key(node_states)
        V_node = self.value(node_states)

        # FIXME: May have to extract the values, project them, and then make a new sparse tensor
        K_edge = self.key(edge_states)
        V_edge = self.value_edge(edge_states)

        edge_indices = edge_states.get_indices()
        Q_masked = self.transpose_for_scores(Q[edge_indices[0], edge_indices[1]])                       # (num_edges, num_heads, att_head_size)
        K_node_masked = self.transpose_for_scores(K_node[edge_indices[0], edge_indices[2]])             # (num_edges, num_heads, att_head_size)
        node2node = torch.sum(Q_masked * K_node_masked, dim=-1) / torch.sqrt(self.attention_head_size)  # (num_edges, num_heads)

        K_edge_masked = self.transpose_for_scores(K_edge[edge_indices[0], edge_indices[2]])
        node2edge = torch.sum(Q_masked * K_edge_masked, dim=-1) / torch.sqrt(self.attention_head_size)  # (num_edges, num_heads)

        logits_val = node2node + node2edge
        logits = torch.sparse_coo_tensor(
            indices=edge_indices,
            values=logits_val,
            size=edge_states.shape[:-1],
            device=logits_val.device,
            is_coalesced=True,
            requires_grad=True
        )
        sparse_attention = torch.sparse_softmax(logits, dim=-1)
        attention_vals = sparse_attention.values()
        attention_vals = self.dropout(attention_vals)

        V_node_masked = V_node[edge_indices[0], edge_indices[2]]
        V_edge_masked = V_edge[edge_indices[0], edge_indices[1], edge_indices[2]]
        V_masked = V_node_masked + V_edge_masked

        elwise_prod = torch.unsqueeze(attention_vals, -1) * V_masked # (num_edges, num_heads, att_head_size)
        elwise_prod = torch.flatten(elwise_prod, start_dim=1) # (num_edges, all_head_size)

        (batch_size, n_nodes) = node_states.shape[:2]
        # (batch * nodes, num_edges)
        edge_mask = torch.sparse_coo_tensor(
            indices=torch.stack([edge_indices[0] * n_nodes + edge_indices[1], torch.arange(edge_indices.shape[1])]),
            values=torch.ones(edge_indices.shape[1]),
            size=(batch_size * n_nodes, edge_indices.shape[1]),
            device=elwise_prod.device,
            is_coalesced=True
        )

        # (batch * nodes, all_head_size)
        flat_node_states = torch.sparse.mm(edge_mask, elwise_prod)
        # (batch, nodes, all_head_size)
        node_states = flat_node_states.reshape(batch_size, n_nodes, -1)

        return node_states

class GatbertAttention(torch.nn.Module):
    def __init__(self, config):
        self.attention = GatbertSelfAttention(config)
        self.output = BertSelfOutput(config)
        pass

    def load_pretrained_weights(self, other: BertAttention):
        self.attention.load_pretrained_weights(other.self)
        self.output.load_state_dict(other.output.state_dict())

    def forward(self, node_states: torch.Tensor, edge_states: torch.Tensor):
        """
        Args:
            node_states: Strided tensor of shape (batch, nodes, hidden_state_size)
            edge_states: Hybrid array of shape (batch, nodes, nodes, hidden_state_size) where last dimension is dense
        """
        node_states = self.attention(node_states, edge_states)
        node_states = self.output(node_states)
        edge_states = self.output(edge_states)
        return (node_states, edge_states)

class GatbertLayer(torch.nn.Module):
    """
    Parallels HF BertLayer class
    """
    def __init__(self, config):
        self.attention = GatbertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    def load_pretrained_weights(self, other: BertLayer):
        self.attention.load_pretrained_weights(other.attention)
        self.intermediate.load_state_dict(other.intermediate.state_dict())
        self.output.load_state_dict(other.output.state_dict())
    def forward(self, node_states: torch.Tensor, edge_states: torch.Tensor):
        node_states = self.attention(node_states, edge_states)

        node_states = self.intermediate(node_states)
        node_states = self.output(node_states)

        # FIXME: May have to extract the values, project, and then make edge_states a sparse tensor again
        edge_states = self.intermediate(edge_states)
        edge_states = self.output(edge_states)

        return (node_states, edge_states)

class GatbertEncoder(torch.nn.Module):
    def __init__(self, config):
        self.layer = torch.nn.ModuleList(GatbertLayer(config) for _ in range(config.num_hidden_layers))
    def load_pretrained_weights(self, other: BertEncoder):
        for (self_layer, other_layer) in zip(self.layer, other.layer):
            self_layer.load_pretrained_weights(other_layer)
    def forward(self, node_states, edge_states):
        for layer_module in self.layer:
            (node_states, edge_states) = layer_module
        return node_states, edge_states

class GatbertEmbeddings(torch.nn.Module):
    def __init__(self, config):
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def load_pretrained_weights(self, other: BertEmbeddings):
        self.word_embeddings.load_state_dict(other.word_embeddings.state_dict())
        self.layer_norm.load_state_dict(other.LayerNorm.state_dict())

    def forward(self, subword_ids: torch.Tensor, pooling_mask: torch.Tensor):
        if not pooling_mask.is_coalesced():
            pooling_mask = pooling_mask.coalesce()
        (batch_size, max_nodes, max_subnodes) = pooling_mask.shape

        # (batch_size, max_subnodes, embedding)
        subnode_embeddings = self.word_embeddings(subword_ids)
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
        node_embeddings = node_embeddings.reshape(batch_size, max_nodes, -1)

        node_embeddings = self.layer_norm(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)

        return node_embeddings

class GatbertModel(torch.nn.Module):
    def __init__(self, config, n_relations: int):
        self.embeddings = GatbertEmbeddings(config)
        self.encoder = GatbertEncoder(config)
        self.relation_embeddings = torch.nn.Embedding(n_relations, config.hidden_size)

    def load_pretrained_weights(self, other: BertModel):
        self.embeddings.load_pretrained_weights(other.embeddings)
        self.encoder.load_pretrained_weights(other.encoder)

    def forward(self, input_ids: torch.Tensor, node_mask: torch.Tensor, edge_indices: torch.Tensor):
        node_states = self.embeddings(input_ids, node_mask)
        (batch_size, n_nodes) = node_states.shape[:2]

        # All the indices save the relation IDs
        simple_edge_indices = edge_indices[:-1]
        relation_indices = edge_indices[-1]

        edge_values = self.relation_embeddings(relation_indices)
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

        return self.encoder(node_states, edge_states)