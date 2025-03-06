import abc
from typing import List, Optional, Literal
import os
import copy
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast, AutoConfig
from transformers.models.bert.modeling_bert import BertConfig
# Local
from .types import Transform
from .gatbert import GatbertModel, GatbertEncoder, GatbertEmbeddings
from .constants import Stance, NodeType, DEFAULT_MODEL
from .config import GatbertConfig
from .encoder import *
from .graph import *
from .cgcn import Cgcn

class StanceClassifier(torch.nn.Module):

    @abc.abstractmethod
    def get_encoder(self) -> Encoder:
        pass

    @abc.abstractmethod
    def load_pretrained_weights(self):
        pass


class BertStanceClassifier(StanceClassifier):
    def __init__(self,
                 pretrained_model: str = DEFAULT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.projection = torch.nn.Linear(
            self.bert.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model))

    def load_pretrained_weights(self):
        # TODO: Find a cleaner way than just reinstantiating the object
        self.bert = BertModel.from_pretrained(self.bert.config.base_model)

    def get_encoder(self):
        return self.__encoder

    def forward(self, *args, **kwargs):
        bert_output = self.bert(*args, **kwargs)
        last_hidden_state = bert_output['last_hidden_state'][:, 0]
        logits = self.projection(last_hidden_state)
        return logits

    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer
        def encode(self, sample: Sample | PretokenizedSample):
            return {
                **encode_text(self.__tokenizer, sample),
                'stance': torch.tensor([sample.stance.value])
            }
        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                **collate_ids(self.__tokenizer, samples, return_attention_mask=True),
                'stance': keyed_scalar_stack(samples, 'stance')
            }


class GatClassifier(StanceClassifier):
    """
    Produces a hidden state summarizing just a graph (no text)
    """
    def __init__(self,
                 pretrained_model: str,
                 graph: os.PathLike,
                 num_layers: int = 2):
        super().__init__()
        graph_obj = CNGraph.read(graph)
        self.config = GatbertConfig(
            BertConfig.from_pretrained(pretrained_model),
            n_relations=len(graph_obj.rel2id),
            num_layers=num_layers,
        )
        self.concept_embeddings = GatbertEmbeddings(self.config)
        self.gat = GatbertEncoder(self.config)
        self.linear = torch.nn.Linear(self.config.hidden_size, len(Stance), bias=False)
        self.__encoder = self.Encoder(graph_obj)
        self.__pretrained_model = pretrained_model

    def load_pretrained_weights(self):
        self.concept_embeddings.load_pretrained_weights(BertModel.from_pretrained(self.__pretrained_model).embeddings)
        # TODO: Load graph embeddings

    def get_encoder(self):
        return self.__encoder

    def forward(self, input_ids, pooling_mask, edge_indices, node_counts):
        # Graph Calculation
        graph_embeddings = self.concept_embeddings(input_ids=input_ids,
                                                   pooling_mask=pooling_mask)
        graph_hidden_states = self.gat(graph_embeddings, edge_indices)
        node_counts = torch.maximum(node_counts, torch.tensor(1))
        avg_graph_hidden_states = torch.sum(graph_hidden_states, dim=1) / torch.unsqueeze(node_counts, dim=-1)
        logits = self.linear(avg_graph_hidden_states)
        return logits

    class Encoder(Encoder):
        def __init__(self, graph: CNGraph):
            self.__graph = graph
        def encode(self, sample: GraphSample) -> TensorDict:
            assert isinstance(sample, GraphSample)
            input_ids = [self.__graph.uri2id[node] for node in sample.kb]
            node_counts = len(input_ids)
            return {
                "input_ids" : torch.tensor([input_ids]),
                "node_counts": torch.tensor([node_counts]),
                "edge_indices": extract_kb_edges(sample),
                "stance": torch.tensor([sample.stance.value])
            }
        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                "input_ids": keyed_pad(samples, 'input_ids'),
                "node_counts": keyed_scalar_stack(samples, "node_counts"),
                "edge_indices": collate_edge_indices(s['edge_indices'] for s in samples),
                "stance": keyed_scalar_stack(samples, "stance")
            }


class HybridClassifier(StanceClassifier):
    def __init__(self,
                pretrained_model: str,
                graph: os.PathLike,
                 config: GatbertConfig):
        super().__init__()

        graph_obj = CNGraph.read(graph)
        self.config = GatbertConfig(
            BertConfig.from_pretrained(pretrained_model),
            n_relations=len(graph_obj.rel2id),
        )
        self.gatbert = GatbertModel(config)
        self.projection = torch.nn.Linear(
            config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), graph_obj)

    def load_pretrained_weights(self):
        self.gatbert.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model))
        # TODO: Load graph node embeddings

    def get_encoder(self):
        return self.__encoder

    def forward(self, *args, **kwargs):
        final_hidden_state = self.gatbert(*args, **kwargs)
        logits = self.projection(final_hidden_state[:, 0])
        return logits


    class Encoder(Encoder):
        def __init__(self,
                     tokenizer: PreTrainedTokenizerFast,
                     graph: CNGraph):
            self.__tokenizer = tokenizer
            self.__graph = graph

        def encode(self, sample: GraphSample):
            assert isinstance(sample, GraphSample)
            tokenizer = self.__tokenizer

            tokenized_text = tokenizer(text=sample.target,
                                       text_pair=sample.context,
                                       is_split_into_words=True,
                                       return_offsets_mapping=True,
                                       return_tensors='pt',
                                       truncation='longest_first')
            device = tokenized_text['input_ids'].device

            kb_input_ids = torch.tensor([[self.__graph.uri2id[node] for node in sample.kb]])
            num_kb_nodes = kb_input_ids.shape[-1]

            # Combine input ids
            concat_ids = torch.concatenate([tokenized_text['input_ids'], kb_input_ids], dim=-1)
            # Add dummy position ids for graph nodes
            position_ids = torch.tensor(
                [i for i in range(tokenized_text['input_ids'].shape[-1])] + \
                [0 for _ in range(kb_input_ids.shape[-1])],
                device=device
            )
            # Add dummy token_type ids for graph nodes
            last_token_type_id = tokenized_text['token_type_ids'][..., -1][0]
            token_type_ids = torch.concatenate([
                tokenized_text['token_type_ids'],
                torch.full_like(kb_input_ids, last_token_type_id)
                ],
                dim=-1
            )

            # old_node_index -> [new_node_indices]
            expand_list = defaultdict(list)
            # For token subwords, we will split a token's nodes into subwords
            # Handle splitting of token nodes into subword nodes
            orig_nodes_index = -1
            for (new_nodes_index, (start, end)) in enumerate(tokenized_text['offset_mapping'].squeeze()):
                if start != end: # Real character, not a special character
                    if start == 0: # Start of a token
                        orig_nodes_index += 1
                    expand_list[orig_nodes_index].append(new_nodes_index)

            orig_text_nodes = len(sample.target) + len(sample.context)
            new_text_nodes = tokenized_text['input_ids'].shape[-1]
            # Indices into a sparse array (batch, max_new_nodes, max_new_nodes, relation)
            # Need a 0 at the beginning for batch
            new_edges = []
            # The original token-to-token edges of a standard BERT model
            new_edges.extend((0, head, tail, TOKEN_TO_TOKEN_RELATION_ID) for (head, tail) in product(range(new_text_nodes), range(new_text_nodes)))
            # The KB edges, with indices adjusted
            max_node_index = orig_text_nodes + num_kb_nodes
            for e in sample.edges:
                if orig_text_nodes <= e.head_node_index < max_node_index:
                    head_list = [new_text_nodes + (e.head_node_index - orig_text_nodes)]
                elif e.head_node_index in expand_list:
                    head_list = expand_list[e.head_node_index]
                else:
                    continue
                if orig_text_nodes <= e.tail_node_index < max_node_index:
                    tail_list = [new_text_nodes + (e.tail_node_index - orig_text_nodes)]
                elif e.tail_node_index in expand_list:
                    tail_list = expand_list[e.tail_node_index]
                else:
                    continue
                new_edges.extend((0, head, tail, e.relation_id) for (head, tail) in product(head_list, tail_list))

            new_edges.sort()
            new_edges = torch.tensor(new_edges, device=device).transpose(1, 0)

            node_type_ids = torch.tensor([NodeType.TOKEN.value] * new_text_nodes + [NodeType.KB.value] * num_kb_nodes, device=device)
            node_type_ids = torch.unsqueeze(node_type_ids, 0)

            return {
                "input_ids" : concat_ids,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "node_type_ids": node_type_ids,
                "edge_indices": new_edges,
                "stance": torch.tensor([sample.stance.value], device=device)
            }
        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                "input_ids": keyed_pad(samples, "input_ids"),
                "position_ids": keyed_pad(samples, "position_ids"),
                "token_type_ids": keyed_pad(samples, "token_type_ids"),
                "node_type_ids": keyed_pad(samples, "node_type_ids"),
                "edge_indices": collate_edge_indices(s['edge_indices'] for s in samples),
                "stance": keyed_scalar_stack(samples, "stance")
            }
    

class ConcatClassifier(StanceClassifier):
    """
    Modeled after https://aclanthology.org/2021.findings-acl.278/,
    except we provide the option to use a GAT instead of a CGCN
    """
    def __init__(self,
                 pretrained_model: str,
                 graph: os.PathLike,
                 graph_model: Literal['cgcn', 'gat'] = 'gat'):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)

        self.entity_embeddings: torch.nn.Embedding = torch.load(get_entity_embeddings(graph))
        self.relation_embeddings: torch.nn.Embedding = torch.load(get_relation_embeddings(graph))
        (_, self.entity_embed_dim) = self.entity_embeddings.weight.shape
        assert len(self.relation_embeddings.weight.shape) == 2
        (self.n_relations, self.relation_embed_dim) = self.relation_embeddings.weight.shape
        assert self.entity_embed_dim == self.relation_embed_dim

        if graph_model == 'gat':
            gat_config = GatbertConfig(
                self.bert.config,
                self.n_relations,
                num_graph_layers=2,
                rel_dims=(self.relation_embed_dim,)
            )
            gat_config.hidden_size = self.entity_embed_dim
            self.gat = GatbertEncoder(gat_config)
        elif graph_model == 'cgcn':
            self.cgcn = Cgcn(self.entity_embed_dim, self.n_relations)
        else:
            raise ValueError(f"Invalid model_type {graph_model}")
        self.model_type = graph_model

        self.pred_head = torch.nn.Linear(2 * self.bert.config.hidden_size + 2 * self.entity_embed_dim, len(Stance), bias=False)
        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), CNGraph.read(graph))
    
    def load_pretrained_weights(self):
        self.concept_embeddings.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model).embeddings)
        # FIXME: Again, need a better approach than just reinstantiating the model
        self.bert = BertModel.from_pretrained(self.config.base_model)

    def get_encoder(self):
        return self.__encoder

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
        target_text_vec = target_text_mask * hidden_states / torch.sum(target_text_mask, dim=-1, keepdim=True)
        context_text_vec = context_text_mask * hidden_states / torch.sum(context_text_mask, dim=-1, keepdim=True)
        # (2) Encode graph
        node_embeddings = self.entity_embeddings(input_ids)
        if self.model_type == 'cgcn':
            rel_embeddings = self.relation_embeddings.weight
            final_node_states, _ = self.cgcn(node_embeddings, edge_indices, rel_embeddings)
        else:
            # TODO: incorporate the relation embeddings later
            final_node_states, _ = self.gat(node_embeddings, edge_indices)
        target_node_vec = target_node_mask * final_node_states / torch.sum(target_node_mask, dim=-1, keepdim=True)
        context_node_vec = context_node_mask * final_node_states / torch.sum(context_node_mask, dim=-1, keepdim=True)
        # (3) CONCAT their representations and project
        feature_vec = torch.concatenate([target_text_vec, context_text_vec, target_node_vec, context_node_vec], dim=-1)
        logits = self.pred_head(feature_vec)
        return logits

    class Encoder(Encoder):
        """
        Creates samples consisting of a graph with only external information (ConceptNet, AMR, etc.)
        and a separate sequence of tokens. The graph and tokens are totally independent.
        """
        def __init__(self, tokenizer: PreTrainedTokenizerFast, graph: CNGraph):
            self.__tokenizer = tokenizer
            self.__graph = graph

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

            input_ids = torch.tensor([[self.__graph.uri2id[node] for node in sample.kb]])

            # FIXME: need a target_text_mask and context_text_mask
            text_encoding = encode_text(self.__tokenizer, sample)

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

            return {
                "text": text_encoding,
                "target_text_mask": target_text_mask,
                "context_text_mask": context_text_mask,

                "input_ids": input_ids,
                "target_node_mask": self.get_target_seeds_mask(sample),
                "context_node_mask": self.get_context_seeds_mask(sample),
                "edge_indices": extract_kb_edges(sample),
                "stance": torch.tensor([sample.stance.value]),
            }
    
        def collate(self, samples: List[Dict[str, TensorDict]]) -> TensorDict:
            rdict = {}

            rdict['text'] = collate_ids(self.__tokenizer, samples, return_attention_mask=True)
            rdict['target_text_mask'] = keyed_pad(samples, 'target_text_mask')
            rdict['context_text_mask'] = keyed_pad(samples, 'context_text_mask')


            rdict['input_ids'] = keyed_pad(samples, 'input_ids')
            rdict['target_node_mask'] = keyed_pad(samples, 'target_node_mask')
            rdict['context_node_mask'] = keyed_pad(samples, 'context_node_mask')
            rdict['node_count'] = keyed_scalar_stack(samples, 'node_count')
            rdict["edge_indices"] = collate_edge_indices(s['edge_indices'] for s in samples)
            rdict["stance"] = keyed_scalar_stack(samples, 'stance')
    
            return rdict

