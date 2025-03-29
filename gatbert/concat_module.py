# STL
import abc
import os
from typing import List, Dict, Tuple
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast
# Local
from .graph_sample import GraphSample
from .base_module import StanceModule
from .constants import DEFAULT_MODEL, Stance, MAX_KB_NODES, SpecialRelation
from .encoder import Encoder, collate_ids, keyed_pad, collate_edge_indices, keyed_scalar_stack, encode_text
from .gatbert import GatbertConfig, GatbertEncoder
from .types import TensorDict
from .cgcn import Cgcn
from .graph import read_entitites, load_kb_embeddings, get_entities_path, get_n_relations


class ConcatModule(StanceModule):

    def __init__(self,
                 graph: os.PathLike,
                 pretrained_model: str = DEFAULT_MODEL,
                 joint_loss: bool = False,
                 num_graph_layers: int = 2,
                 pretrained_relations: bool = False):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__()
        self.save_hyperparameters()

        self.bert = BertModel.from_pretrained(pretrained_model)

        self.entity_embeddings, self.relation_embeddings = load_kb_embeddings(graph, pretrained_relations)
        (_, self.entity_embed_dim) = self.entity_embeddings.weight.shape
        (self.n_relations, self.relation_embed_dim) = self.relation_embeddings.weight.shape

        self.graph_head = torch.nn.Linear(2 * self.entity_embed_dim,        len(Stance), bias=False)
        self.text_head  = torch.nn.Linear(2 * self.bert.config.hidden_size, len(Stance), bias=False)
        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), graph)

    @abc.abstractmethod
    def _graph_encode(self, node_states, edge_indices) -> torch.Tensor:
        """
        Returns (node_states, edge_states)
        """

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    @staticmethod
    def masked_average(mask, embeddings) -> torch.Tensor:
        # Need the epsilon to prevent divide-by-zero errors
        denom = torch.sum(mask, dim=-1, keepdim=True) + 1e-6
        return torch.sum(torch.unsqueeze(mask, -1) * embeddings, dim=-2) / denom

    def training_step(self, batch, batch_idx):
        labels = batch.pop("stance")
        # Calls the forward method defined in subclass
        logits, text_logits, graph_logits = self(**batch)

        joint_loss = torch.nn.functional.cross_entropy(logits, labels)
        if self.hparams.joint_loss:
            text_loss = torch.nn.functional.cross_entropy(text_logits, labels)
            graph_loss = torch.nn.functional.cross_entropy(graph_logits, labels)
            self.log("loss_text", text_loss)
            self.log("loss_graph", graph_loss)
            self.log("loss_joint", joint_loss)
            total_loss = text_loss + graph_loss + joint_loss
        else:
            total_loss = joint_loss

        self.log("loss", total_loss)
        return total_loss

    def forward(self,
                text,
                target_text_mask,
                context_text_mask,
                input_ids,
                target_node_mask,
                context_node_mask,
                edge_indices):


        # (1) Encode text
        bert_out = self.bert(**text)
        hidden_states = bert_out.last_hidden_state
        target_text_vec = self.masked_average(target_text_mask, hidden_states)
        context_text_vec = self.masked_average(context_text_mask, hidden_states)
        # (2) Encode graph
        node_embeddings = self.entity_embeddings(input_ids)
        final_node_states = self._graph_encode(node_embeddings, edge_indices)
        target_node_vec = self.masked_average(target_node_mask, final_node_states)
        context_node_vec = self.masked_average(context_node_mask, final_node_states)

        text_feature_vec = torch.concatenate([target_text_vec, context_text_vec], dim=-1)
        graph_feature_vec = torch.concatenate([target_node_vec, context_node_vec], dim=-1)

        text_logits = self.text_head(text_feature_vec)
        graph_logits = self.graph_head(graph_feature_vec)
        logits = text_logits + graph_logits
        return logits, text_logits, graph_logits

    class Encoder(Encoder):
        """
        Creates samples consisting of a graph with only external information (ConceptNet, AMR, etc.)
        and a separate sequence of tokens. The graph and tokens are totally independent.
        """
        def __init__(self, tokenizer: PreTrainedTokenizerFast, graph: os.PathLike):
            self.__tokenizer = tokenizer
            self.__uri2id = read_entitites(get_entities_path(graph))
            self.__total_relations = get_n_relations(graph) + len(SpecialRelation)

        @staticmethod
        def get_target_seeds_mask(sample: GraphSample) -> torch.Tensor:
            mask = torch.zeros([1, len(sample.kb)], dtype=torch.bool)
            text_end = len(sample.target) + len(sample.context)
            target_len = len(sample.target)
            target_seeds = [
                e.tail_node_index - text_end
                for e in sample.edges
                if e.head_node_index < target_len and e.tail_node_index >= text_end
            ] + [
                e.head_node_index - text_end
                for e in sample.edges
                if e.tail_node_index < target_len and e.head_node_index >= text_end
            ]
            mask[0, target_seeds] = True
            return mask

        @staticmethod
        def get_context_seeds_mask(sample: GraphSample) -> torch.Tensor:
            mask = torch.zeros([1, len(sample.kb)], dtype=torch.bool)
            context_start = len(sample.target)
            text_end = len(sample.target) + len(sample.context)
            context_seeds = [
                e.tail_node_index - text_end
                for e in sample.edges
                if context_start <= e.head_node_index < text_end and e.tail_node_index >= text_end
            ] + [
                e.head_node_index - text_end
                for e in sample.edges
                if context_start <= e.tail_node_index < text_end and e.head_node_index >= text_end
            ]
            mask[0, context_seeds] = True
            return mask

    
        def encode(self, sample: GraphSample):
            assert isinstance(sample, GraphSample)

            input_ids = torch.tensor([[self.__uri2id[node] for node in sample.kb[:MAX_KB_NODES]]], dtype=torch.int64)
            num_kb_nodes = input_ids.shape[-1]
            orig_text_nodes = len(sample.target) + len(sample.context)
            # Only keep edges between two graph concepts
            iter_edge = filter(lambda e: e.head_node_index >= orig_text_nodes and e.tail_node_index >= orig_text_nodes, sample.edges)
            iter_edge = map(lambda e: (0, e.head_node_index - orig_text_nodes, e.tail_node_index - orig_text_nodes, e.relation_id), iter_edge)
            # Filter out edges pointing to truncated nodes
            iter_edge = filter(lambda e: e[1] < num_kb_nodes and e[2] < num_kb_nodes, iter_edge)
            # Handle negative relation indices
            iter_edge = map(lambda e: (*e[:-1], e[-1] % self.__total_relations), iter_edge)
            edge_indices = sorted(iter_edge)
            if edge_indices:
                edge_indices = torch.tensor(edge_indices).transpose(1, 0)
            else:
                edge_indices = torch.empty([4, 0], dtype=torch.int)

            text_encoding = encode_text(self.__tokenizer, sample, tokenizer_kwargs={"return_special_tokens_mask": True})

            special_tokens_mask = text_encoding.pop('special_tokens_mask')
            special_inds = torch.where(special_tokens_mask)[-1]

            seqlen = text_encoding['input_ids'].numel()
            cls_ind = special_inds[0]
            sep_ind = special_inds[1]
            if len(special_inds) > 2:
                end_ind = special_inds[2]
            else:
                end_ind = seqlen
            all_inds = torch.arange(0, seqlen)
            target_text_mask = torch.logical_and(cls_ind < all_inds, all_inds < sep_ind)
            context_text_mask = torch.logical_and(sep_ind < all_inds, all_inds < end_ind)

            target_node_mask = self.get_target_seeds_mask(sample)[..., :num_kb_nodes]
            context_node_mask = self.get_context_seeds_mask(sample)[..., :num_kb_nodes]
            return {
                "text": text_encoding,
                "target_text_mask": target_text_mask,
                "context_text_mask": context_text_mask,

                "input_ids": input_ids,
                "target_node_mask": target_node_mask,
                "context_node_mask": context_node_mask,
                "edge_indices": edge_indices,
                "stance": torch.tensor([sample.stance.value]),
            }
    
        def collate(self, samples: List[Dict[str, TensorDict]]) -> TensorDict:
            rdict = {}

            rdict['text'] = collate_ids(self.__tokenizer, [s['text'] for s in samples], return_attention_mask=True)
            rdict['target_text_mask'] = keyed_pad(samples, 'target_text_mask')
            rdict['context_text_mask'] = keyed_pad(samples, 'context_text_mask')


            rdict['input_ids'] = keyed_pad(samples, 'input_ids')
            rdict['target_node_mask'] = keyed_pad(samples, 'target_node_mask')
            rdict['context_node_mask'] = keyed_pad(samples, 'context_node_mask')
            rdict["edge_indices"] = collate_edge_indices(s['edge_indices'] for s in samples)
            rdict["stance"] = keyed_scalar_stack(samples, 'stance')
    
            return rdict


class ConcatCgcnModule(ConcatModule):
    def __init__(self, *parent_args, **parent_kwargs):
        super().__init__(*parent_args, **parent_kwargs)
        self.cgcn = Cgcn(self.entity_embed_dim, self.n_relations, n_layers=self.hparams.num_graph_layers)

    def _graph_encode(self, node_states, edge_indices):
        rel_embeddings = self.relation_embeddings.weight
        final_node_states, _ = self.cgcn(node_states, edge_indices, rel_embeddings)
        return final_node_states

class ConcatGatModule(ConcatModule):
    def __init__(self, *parent_args, **parent_kwargs):
        super().__init__(*parent_args, **parent_kwargs)
        gat_config = GatbertConfig(
            self.bert.config,
            self.n_relations,
            num_graph_layers=self.hparams.num_graph_layers,
            rel_dims=(self.relation_embed_dim,)
        )
        gat_config.num_attention_heads = 1
        gat_config.hidden_size = self.entity_embed_dim
        self.gat = GatbertEncoder(gat_config)
    def _graph_encode(self, node_states, edge_indices):
        # TODO: incorporate the relation embeddings later
        final_node_states, _ = self.gat(node_states, edge_indices)
        return final_node_states
