import abc
import inspect
from typing import List, Optional
import os
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
    def __init__(self,
                 config: GatbertConfig,
                 graph: os.PathLike):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__(config)

        self.bert = BertModel(config.wrapped)
        self.concept_embeddings = GatbertEmbeddings(config)
        self.gat = GatbertEncoder(config)

        self.linear = torch.nn.Linear(config.hidden_size + self.bert.config.hidden_size, len(Stance), bias=False)
    
    def load_pretrained_weights(self):
        self.concept_embeddings.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model).embeddings)
        # FIXME: Again, need a better approach than just reinstantiating the model
        self.bert = BertModel.from_pretrained(self.config.base_model)

    def forward(self, text, graph):
        # Text Calculation
        bert_out = self.bert(**text)
        text_vec = bert_out.last_hidden_state[:, 0]
        # Graph Calculation
        edge_indices = graph.pop('edge_indices')
        node_counts = graph.pop('node_counts')
        node_counts = torch.maximum(node_counts, torch.tensor(1))
        graph_embeddings = self.concept_embeddings(**graph)
        graph_hidden_states = self.gat(graph_embeddings, edge_indices)
        avg_graph_hidden_states = torch.sum(graph_hidden_states, dim=1) / torch.unsqueeze(node_counts, dim=-1)

        # Concat
        feature_vec = torch.concat([text_vec, avg_graph_hidden_states], dim=-1)

        logits = self.linear(feature_vec)
        return logits

    class Encoder(Encoder):
        """
        Creates samples consisting of a graph with only external information (ConceptNet, AMR, etc.)
        and a separate sequence of tokens. The graph and tokens are totally independent.
        """
        def __init__(self, tokenizer: PreTrainedTokenizerFast, graph: CNGraph):
            self.__tokenizer = tokenizer
            self.__graph = graph
    
        def encode(self, sample: GraphSample):
            assert isinstance(sample, GraphSample)

            input_ids = torch.tensor([[self.__graph.uri2id[node] for node in sample.kb]])
            node_type_ids = torch.full_like(input_ids, fill_value=NodeType.KB)

            # FIXME: Need node counts here

            return {
                "text": encode_text(self.__tokenizer, sample),

                "input_ids": input_ids,
                "node_type_ids": node_type_ids,
                "edge_indices": extract_kb_edges(sample),
                "stance": torch.tensor([sample.stance.value]),
            }
    
        def collate(self, samples: List[Dict[str, TensorDict]]) -> TensorDict:
            rdict = {}

            rdict['text'] = collate_ids(self.__tokenizer, samples, return_attention_mask=True)

            rdict['input_ids'] = keyed_pad(samples, 'input_ids')
            rdict["node_type_ids"] = keyed_pad(samples, 'node_type_ids')
            rdict["edge_indices"] = collate_edge_indices(s['edge_indices'] for s in samples)

            rdict["stance"] = keyed_scalar_stack(samples, 'stance')
    
            return rdict

