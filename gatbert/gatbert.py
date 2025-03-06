# STL
from typing import Optional
# 3rd Party
import torch
from transformers.models.bert.modeling_bert import \
    BertSelfOutput, \
    BertIntermediate, \
    BertOutput, \
    BertAttention, \
    BertLayer, \
    BertEncoder, \
    BertEmbeddings, \
    BertModel
# Local
from .self_attention import GatbertSelfAttention, \
    EdgeAsAttendeeSelfAttention, \
    TranslatedKeySelfAttention, \
    RelationInnerProdSelfAttention, \
    HeterogeneousSelfAttention
from .rel_mat_embeddings import RelationMatrixEmbeddings
from .config import GatbertConfig
from .utils import prod

class GatbertAttention(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        super().__init__()
        if config.att_type == "edge_as_att":
            self_attention = EdgeAsAttendeeSelfAttention(config)
        elif config.att_type == "trans_key":
            self_attention = TranslatedKeySelfAttention(config)
        elif config.att_type == "rel_mat":
            raise ValueError("Temporarily not supported")
            relation_embeddings = RelationMatrixEmbeddings(config.n_relations, config.hidden_size // config.num_attention_heads)
            attention_factory = lambda i: RelationInnerProdSelfAttention(config, relation_embeddings)
        elif config.att_type == "hetero":
            raise ValueError("Temporarily not supported")
            relation_embeddings = RelationMatrixEmbeddings(config.n_relations, config.hidden_size // config.num_attention_heads)
            attention_factory = lambda i: HeterogeneousSelfAttention(config, relation_embeddings)
        else:
            raise ValueError(f"Invalid attention type {config.att_type}")

        self.attention = self_attention
        self.output = BertSelfOutput(config)

    def load_pretrained_weights(self, other: BertAttention):
        self.attention.load_pretrained_weights(other.self)
        self.output.load_state_dict(other.output.state_dict())

    def forward(self, hidden_states: torch.Tensor, edge_indices: torch.Tensor,
                node_type_ids: torch.Tensor, relation_states: Optional[torch.Tensor] = None):
        """
        Args:
            node_states: Strided tensor of shape (batch, nodes, hidden_state_size)
            edge_states: Hybrid array of shape (batch, nodes, nodes, hidden_state_size) where last dimension is dense
        """
        self_outputs = self.attention(hidden_states, edge_indices, node_type_ids=node_type_ids, rel_states=relation_states)
        outputs = self.output(self_outputs, hidden_states)

        return outputs

class GatbertLayer(torch.nn.Module):
    """
    Parallels HF BertLayer class
    """
    def __init__(self, config: GatbertConfig):
        super().__init__()
        self.attention = GatbertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # TODO: Only instantiate this when we know it will get used
        self.rel_dims = config.rel_dims
        total_rel_dims = prod(self.rel_dims)
        self.rel_proj = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=-len(self.rel_dims)),
            torch.nn.Linear(total_rel_dims, total_rel_dims, bias=False),
            torch.nn.Unflatten(-1, self.rel_dims)
        )

    def load_pretrained_weights(self, other: BertLayer):
        self.attention.load_pretrained_weights(other.attention)
        self.intermediate.load_state_dict(other.intermediate.state_dict())
        self.output.load_state_dict(other.output.state_dict())
    def forward(self, node_states: torch.Tensor, edge_indices: torch.Tensor,
                node_type_ids: torch.Tensor, relation_states: Optional[torch.Tensor] = None):

        node_attention_output = self.attention(node_states, edge_indices, node_type_ids=node_type_ids, relation_states=relation_states)
        new_node_states = self.intermediate(node_attention_output)
        new_node_states = self.output(new_node_states, node_attention_output)

        new_relation_states = self.rel_proj(relation_states) if relation_states else None

        return new_node_states, new_relation_states

class GatbertEncoder(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        super().__init__()
        self.layer = torch.nn.ModuleList(GatbertLayer(config) for _ in range(config.num_graph_layers))

    def load_pretrained_weights(self, other: BertEncoder):
        if len(self.layer) != len(other.layer):
            raise Warning(f"Trying to initialize {len(self.layer)} GatbertLayers from only {len(other.layer)} BertLayers")
        for (self_layer, other_layer) in zip(self.layer, other.layer):
            self_layer.load_pretrained_weights(other_layer)
    def forward(self, node_states, edge_indices: torch.Tensor,
                node_type_ids: Optional[torch.Tensor] = None, relation_states: Optional[torch.Tensor] = None):
        if node_type_ids is None:
            node_type_ids = torch.zeros(node_states.shape[:2], dtype=torch.int, device=node_states.device)

        for layer_module in self.layer:
            node_states, relation_states = layer_module(node_states, edge_indices, node_type_ids=node_type_ids, relation_states=relation_states)
        return node_states, relation_states

class GatbertEmbeddings(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.hidden_size = config.hidden_size

    def load_pretrained_weights(self, other: BertEmbeddings):
        self.word_embeddings.load_state_dict(other.word_embeddings.state_dict())
        self.position_embeddings.load_state_dict(other.position_embeddings.state_dict())
        self.token_type_embeddings.load_state_dict(other.token_type_embeddings.state_dict())
        self.layer_norm.load_state_dict(other.LayerNorm.state_dict())

    def forward(self,
                input_ids: torch.Tensor,
                pooling_mask: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None):
        if not pooling_mask.is_coalesced():
            pooling_mask = pooling_mask.coalesce()
        (batch_size, max_nodes, max_subnodes) = pooling_mask.shape

        # (batch_size, max_subnodes, embedding)
        subnode_embeddings = self.word_embeddings(input_ids)
        if position_ids is not None:
            pos_embed = self.position_embeddings(position_ids)
            subnode_embeddings += pos_embed
        if token_type_ids is not None:
            tok_type_embed = self.token_type_embeddings(token_type_ids)
            subnode_embeddings += tok_type_embed
        # (batch_size * max_subnodes, embedding)
        subnode_embeddings = torch.flatten(subnode_embeddings, end_dim=-2)


        indices = pooling_mask.indices()
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
        node_embeddings = node_embeddings.reshape(batch_size, max_nodes, self.hidden_size)

        node_embeddings = self.layer_norm(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)

        return node_embeddings

class GatbertModel(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        raise ValueError("Need to drop use of this class")
        super().__init__()
        self.embeddings = GatbertEmbeddings(config)
        self.encoder = GatbertEncoder(config)
        # self.relation_embeddings = torch.nn.Embedding(n_relations, config.hidden_size)

    def load_pretrained_weights(self, other: BertModel):
        self.embeddings.load_pretrained_weights(other.embeddings)
        self.encoder.load_pretrained_weights(other.encoder)

    def forward(self,
                input_ids: torch.Tensor,
                pooling_mask: torch.Tensor,
                edge_indices: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                node_type_ids: Optional[torch.Tensor] = None):
        node_states = self.embeddings(input_ids, pooling_mask, position_ids=position_ids, token_type_ids=token_type_ids)

        return self.encoder(node_states, edge_indices, node_type_ids=node_type_ids) #, edge_states.values())