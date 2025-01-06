# STL

# 3rd Party
import torch
from transformers import BertModel
# Local
from .constants import NODE_PAD_ID, NodeType
from .rgat_layer import RGATLayer


class GATBert(torch.nn.Module):
    def __init__(self,
                 pretrained_model: str,
                 n_relations: int,
                 n_kb_nodes: int,
                 n_classes: int,
                 n_bases: int = 20,
                 ):
        """
        Args:
            n_kb_nodes: Number of unique kb nodes (including the padding node)
            n_relation: Number of unique relations (including the padding relation)

        Relation 0 represents token-to-token
        """
        super().__init__()
        self.__n_relations = n_relations
        self.__bert_model = BertModel.from_pretrained(pretrained_model)
        self.__feature_size: int = self.__bert_model.config.hidden_size
        self.__n_classes = n_classes

        self.__pad_token_id: int = self.__bert_model.config.pad_token_id

        self.__kb_embeddings = torch.nn.Embedding(
            num_embeddings=n_kb_nodes,
            embedding_dim=self.__feature_size,
            padding_idx=NODE_PAD_ID
        )

        self.__rgat = RGATLayer(
            in_features=self.__feature_size,
            attention_units=self.__feature_size,
            out_features=self.__feature_size,
            n_heads=6,
            n_relations=self.__n_relations,
            n_bases=n_bases,
            attention_mode='wirgat'
        )

        self.__projection = torch.nn.Linear(self.__feature_size, self.__n_classes)

    def forward(self,
                 kb_ids: torch.Tensor,
                 edge_indices: torch.Tensor,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 **bert_kwargs):
        """

        Args:
            kb_ids: (batch, max_external_nodes) array with the IDs of knowledge base nodes
            edge_indices: (?, 6) index array with latter 6 components being (batch, head, tail, relation, head_node_type, tail_node_type)
                batch indexes into the batch dim of other arguments
                head indexes into the seq dim of input_ids, or the node dim of kb_ids
                tail indexes into the seq dim of input_ids, or the node dim of kb_ids
                head_node_type indicates whether head indexes into input_ids or kb_ids
                tail_node_type indicates whether tail indexes into input_ids or kb_ids
            input_ids: (batch, max_sequence_length) array with token IDs
                Indexes into our token embeddings
            attention_mask: attention mask array for input_ids
            bert_kwargs: kwargs to pass on to underlying bert model
        Returns:
            (batch, 3) array of stance class log probabilities
        """

        bert_encodings: torch.Tensor = self.__bert_model(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             **bert_kwargs).last_hidden_state
        node_embeddings = self.__kb_embeddings(kb_ids)

        not_token_padding = input_ids != self.__pad_token_id
        # Number of tokens in each sequence of the batch.
        n_tokens = torch.sum(not_token_padding, dim=-1)
        not_node_padding = kb_ids != NODE_PAD_ID
        # Number of external nodes in each sequence of the batch
        n_kb_nodes = torch.sum(not_node_padding, dim=-1)
        max_graph_size = torch.max(n_tokens + n_kb_nodes)

        # Pack the features from tokens and external KB nodes together into one feature map
        # All features are zero by default
        node_features = torch.zeros([input_ids.shape[0], max_graph_size, self.__feature_size])
        # The first nodes' features should be from tokens
        batch_token_indices, token_indices = torch.where(not_token_padding)
        node_features[batch_token_indices, token_indices] = bert_encodings[batch_token_indices, token_indices]
        # The immediately subsequent nodes should have KB features
        batch_node_indices, node_indices = torch.where(not_node_padding)
        translated_node_indices = n_tokens[batch_node_indices] + node_indices
        node_features[batch_node_indices, translated_node_indices] += node_embeddings[batch_node_indices, node_indices]
        # Everything else following is zeros, since we used torch.zeros() to construct node_embeddings

        # The GAT layer doesn't take node types--it expects a unique ID for each node
        # Thus, we take the number of token nodes for a sequence, and use that as an
        # offset for the IDs of KB nodes
        kb_head_indices = torch.where(edge_indices[4] == NodeType.KB.value)[0]
        head_offsets = n_tokens[edge_indices[0, kb_head_indices]]
        edge_indices[1, kb_head_indices] += head_offsets

        kb_tail_indices = torch.where(edge_indices[5] == NodeType.KB.value)[0]
        tail_offsets = n_tokens[edge_indices[0, kb_tail_indices]]
        edge_indices[2, kb_tail_indices] += tail_offsets

        edges_tensor = torch.sparse_coo_tensor(
            indices=edge_indices[:4],
            values=torch.ones(edge_indices.shape[1]),
            size=(input_ids.shape[0], max_graph_size, max_graph_size, self.__n_relations),
            is_coalesced=True
        )

        last_node_states = self.__rgat(node_features, edges_tensor)
        # The first node will correspond to the first token of the sequence, which is [CLS] for Bert models
        cls_node_features = last_node_states[:, 0]
        logits = self.__projection(cls_node_features)
        return logits