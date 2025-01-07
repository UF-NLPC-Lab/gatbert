# STL
from typing import Literal
# 3rd Party
import torch
import lightning as L
from transformers import AutoModel
# Local
from .f1_calc import F1Calc
from .constants import NODE_PAD_ID, NodeType, Stance, DEFAULT_MODEL
from .rgat_layer import RGATLayer

class StanceModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.__ce = torch.nn.CrossEntropyLoss()
        self.__calc = F1Calc()

    def configure_optimizers(self):
        return torch.optim.Adam()
    def training_step(self, batch, batch_idx):
        labels = batch.pop("stance")
        # Calls the forward method defined in subclass
        logits = self(**batch)
        loss = self.__ce(logits, labels)
        self.log("loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        self.__eval_step(batch, batch_idx)
    def test_step(self, batch, batch_idx):
        self.__eval_step(batch, batch_idx)
    def on_validation_epoch_end(self):
        self.__eval_finish('val')
    def on_test_epoch_end(self):
        self.__eval_finish('test')


    def __eval_step(self, batch, batch_idx):
        labels = batch.pop('stance').view(-1)
        logits = self(**batch)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        self.__calc.record(probs, labels)
    def __eval_finish(self, stage):
        self.__log_stats(self.__calc, f"{stage}")
    def __log_stats(self, calc: F1Calc, prefix):
        calc.summarize()
        self.log(f"{prefix}_favor_precision", calc.favor_precision)
        self.log(f"{prefix}_favor_recall", calc.favor_recall)
        self.log(f"{prefix}_favor_f1", calc.favor_f1)
        self.log(f"{prefix}_against_precision", calc.against_precision)
        self.log(f"{prefix}_against_recall", calc.against_recall)
        self.log(f"{prefix}_against_f1", calc.against_f1)
        self.log(f"{prefix}_macro_f1", calc.macro_f1)
        calc.reset()

class GATBert(StanceModule):
    def __init__(self,
                 n_relations: int,
                 n_kb_nodes: int,
                 n_heads: int = 6,
                 n_bases: int = 20,
                 attention_mode: Literal['wirgat', 'argat'] = 'argat',
                 pretrained_model: str = DEFAULT_MODEL
                 ):
        """
        Args:
            n_kb_nodes: Number of unique kb nodes (including the padding node)
            n_relation: Number of unique relations (including the padding relation)

        Relation 0 represents token-to-token
        """
        super().__init__()
        self.save_hyperparameters()
        self.__n_relations = n_relations
        self.__bert_model = AutoModel.from_pretrained(pretrained_model)
        self.__feature_size: int = self.__bert_model.config.hidden_size
        self.__n_classes = len(Stance)
        self.__n_heads = n_heads
        self.__n_bases = n_bases
        self.__attention_mode = attention_mode

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
            n_heads=self.__n_heads,
            n_relations=self.__n_relations,
            n_bases=self.__n_bases,
            attention_mode=self.__attention_mode
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
        node_features = torch.zeros([input_ids.shape[0], max_graph_size, self.__feature_size], device=bert_encodings.device)
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
            is_coalesced=True,
            device=edge_indices.device
        )

        last_node_states = self.__rgat(node_features, edges_tensor)
        # The first node will correspond to the first token of the sequence, which is [CLS] for Bert models
        cls_node_features = last_node_states[:, 0]
        logits = self.__projection(cls_node_features)
        return logits