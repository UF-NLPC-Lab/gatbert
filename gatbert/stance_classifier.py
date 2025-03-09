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

class BertClassifier(StanceClassifier):
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


class HybridClassifier(StanceClassifier):
    def __init__(self,
                graph: os.PathLike,
                pretrained_model: str = DEFAULT_MODEL,
                ):
        super().__init__()

        graph_obj = CNGraph.read(graph)
        self.config = GatbertConfig(
            BertConfig.from_pretrained(pretrained_model),
            n_relations=len(graph_obj.rel2id),
        )

        self.embeddings = GatbertEmbeddings(self.config, graph)
        self.encoder = GatbertEncoder(self.config)
        self.encoder.load_pretrained_weights(BertModel.from_pretrained(pretrained_model).encoder)
        self.projection = torch.nn.Linear(
            self.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
        self.__preprocessor = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), graph_obj)

    def get_encoder(self):
        return self.__preprocessor

    def forward(self, input_ids, kb_mask, edge_indices, position_ids = None, token_type_ids = None):
        node_embeddings = self.embeddings(input_ids=input_ids,
                                     kb_mask=kb_mask,
                                     position_ids=position_ids,
                                     token_type_ids=token_type_ids)
        # TODO: incorporate relational embeddings ?
        final_node_states, _ = self.encoder(node_embeddings, edge_indices=edge_indices)
        logits = self.projection(final_node_states[:, 0])
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

            kb_input_ids = torch.tensor([[self.__graph.uri2id[node] for node in sample.kb[:MAX_KB_NODES]]], dtype=torch.int64)
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

            kb_mask = torch.tensor([0] * new_text_nodes + [1] * num_kb_nodes, device=device)
            kb_mask = torch.unsqueeze(kb_mask, 0)

            return {
                "input_ids" : concat_ids,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "kb_mask": kb_mask,
                "edge_indices": new_edges,
                "stance": torch.tensor([sample.stance.value], device=device)
            }
        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                "input_ids": keyed_pad(samples, "input_ids"),
                "position_ids": keyed_pad(samples, "position_ids"),
                "token_type_ids": keyed_pad(samples, "token_type_ids"),
                "kb_mask": keyed_pad(samples, "kb_mask"),
                "edge_indices": collate_edge_indices(s['edge_indices'] for s in samples),
                "stance": keyed_scalar_stack(samples, "stance")
            }
    

class ConcatClassifier(StanceClassifier):
    """
    Modeled after https://aclanthology.org/2021.findings-acl.278/,
    except we provide the option to use a GAT instead of a CGCN
    """
    def __init__(self,
                 graph: os.PathLike,
                 pretrained_model: str = DEFAULT_MODEL,
                 graph_model: Literal['cgcn', 'gat'] = 'gat',
                 num_graph_layers: int = 2):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)

        self.entity_embeddings: torch.nn.Embedding = torch.load(get_entity_embeddings(graph), weights_only=False)
        self.relation_embeddings: torch.nn.Embedding = torch.load(get_relation_embeddings(graph), weights_only=False)
        (_, self.entity_embed_dim) = self.entity_embeddings.weight.shape
        assert len(self.relation_embeddings.weight.shape) == 2
        (self.n_relations, self.relation_embed_dim) = self.relation_embeddings.weight.shape
        assert self.entity_embed_dim == self.relation_embed_dim

        if graph_model == 'gat':
            gat_config = GatbertConfig(
                self.bert.config,
                self.n_relations,
                num_graph_layers=num_graph_layers,
                rel_dims=(self.relation_embed_dim,)
            )
            gat_config.num_attention_heads = 1
            gat_config.hidden_size = self.entity_embed_dim
            self.gat = GatbertEncoder(gat_config)
        elif graph_model == 'cgcn':
            self.cgcn = Cgcn(self.entity_embed_dim, self.n_relations, n_layers=num_graph_layers)
        else:
            raise ValueError(f"Invalid model_type {graph_model}")
        self.model_type = graph_model

        self.pred_head = torch.nn.Linear(2 * self.bert.config.hidden_size + 2 * self.entity_embed_dim, len(Stance), bias=False)
        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), CNGraph.read(graph))
    
    def get_encoder(self):
        return self.__encoder

    @staticmethod
    def masked_average(mask, embeddings) -> torch.Tensor:
        # Need the epsilon to prevent divide-by-zero errors
        denom = torch.sum(mask, dim=-1, keepdim=True) + 1e-6
        return torch.sum(torch.unsqueeze(mask, -1) * embeddings, dim=-2) / denom

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
        if self.model_type == 'cgcn':
            rel_embeddings = self.relation_embeddings.weight
            final_node_states, _ = self.cgcn(node_embeddings, edge_indices, rel_embeddings)
        else:
            # TODO: incorporate the relation embeddings later
            final_node_states, _ = self.gat(node_embeddings, edge_indices)
        target_node_vec = self.masked_average(target_node_mask, final_node_states)
        context_node_vec = self.masked_average(context_node_mask, final_node_states)
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

            # FIXME: Need to trim to :MAX_KB_NODES here as well
            input_ids = torch.tensor([[self.__graph.uri2id[node] for node in sample.kb]], dtype=torch.int64)

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

            target_node_mask = self.get_target_seeds_mask(sample)
            context_node_mask = self.get_context_seeds_mask(sample)
            return {
                "text": text_encoding,
                "target_text_mask": target_text_mask,
                "context_text_mask": context_text_mask,

                "input_ids": input_ids,
                "target_node_mask": target_node_mask,
                "context_node_mask": context_node_mask,
                "edge_indices": extract_kb_edges(sample),
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

