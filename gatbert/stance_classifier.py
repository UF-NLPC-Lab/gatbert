import abc
import inspect
from typing import List, Optional
# 3rd Party
import torch
from transformers import BertModel, PreTrainedTokenizerFast
from transformers.models.bert.modeling_bert import BertConfig
# Local
from .types import Transform
from .gatbert import GatbertModel, GatbertEncoder, GatbertEmbeddings
from .constants import Stance
from .config import GatbertConfig
from .encoder import *

class StanceClassifier(torch.nn.Module):
    def __init__(self, config: GatbertConfig):
        super().__init__()
        self.config = config
        pass

    @abc.abstractmethod
    def load_pretrained_weights(self):
        pass

    @classmethod
    def get_encoder(cls, tokenizer: PreTrainedTokenizerFast, transforms: Optional[List[Transform]] = None) -> Encoder:
        # Because I didn't feel like adding a second argument to these other classes
        constructor = cls.Encoder
        params = inspect.signature(constructor).parameters
        kwargs = {}
        if "transforms" in params:
            kwargs['transforms'] = transforms
        return constructor(tokenizer, **kwargs)

class ExternalClassifier(StanceClassifier):
    """
    Produces a hidden state summarizing just a graph (no text)
    """
    def __init__(self, config: GatbertConfig):
        super().__init__(config)
        self.concept_embeddings = GatbertEmbeddings(config)
        self.gat = GatbertEncoder(config)
        self.linear = torch.nn.Linear(config.hidden_size, len(Stance), bias=False)

    def load_pretrained_weights(self):
        self.concept_embeddings.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model).embeddings)

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
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer

        def encode(self, sample: GraphSample) -> TensorDict:
            assert isinstance(sample, GraphSample)
            input_ids, pool_inds = encode_kb_nodes(self.__tokenizer, sample.kb)
            mask_indices, mask_values = build_average_pool_mask(pool_inds)

            mask_indices = [(0, *index) for index in mask_indices] # Prepend batch dim

            device = input_ids.device
            node_mask = torch.sparse_coo_tensor(
                indices=torch.tensor(mask_indices, device=device).transpose(1, 0),
                values=torch.tensor(mask_values, device=device),
                size=(1, len(sample.kb), input_ids.shape[-1]),
                is_coalesced=True,
                requires_grad=True,
                dtype=torch.float,
                device=device
            )

            orig_text_nodes = len(sample.target) + len(sample.context)
            # Only keep edges between two graph concepts
            iter_edge = filter(lambda e: e.head_node_index >= orig_text_nodes and e.tail_node_index >= orig_text_nodes, sample.edges)
            iter_edge = map(lambda e: (0, e.head_node_index - orig_text_nodes, e.tail_node_index - orig_text_nodes, e.relation_id), iter_edge) 
            edge_indices = sorted(iter_edge)
            if edge_indices:
                edge_indices = torch.tensor(edge_indices, device=device).transpose(1, 0)
            else:
                edge_indices = torch.empty([4, 0], dtype=torch.int, device=device)
            return {
                "input_ids" : input_ids,
                "pooling_mask" : node_mask,
                "edge_indices": edge_indices,
                "stance": torch.tensor(sample.stance.value, device=device)
            }

        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {
                **collate_ids(self.__tokenizer, samples),
                **collate_graph_data(samples, return_node_counts=True),
                'stance': torch.stack([s['stance'] for s in samples])
            }


class HybridClassifier(StanceClassifier):
    def __init__(self, config: GatbertConfig):
        super().__init__(config)
        self.gatbert = GatbertModel(config)
        self.projection = torch.nn.Linear(
            config.hidden_size,
            out_features=len(Stance),
            bias=False
        )

    def load_pretrained_weights(self):
        self.gatbert.load_pretrained_weights(BertModel.from_pretrained(self.config.base_model))

    def forward(self, *args, **kwargs):
        final_hidden_state = self.gatbert(*args, **kwargs)
        logits = self.projection(final_hidden_state[:, 0])
        return logits


    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast, transforms: Optional[List[Transform]] = None):
            self.__tokenizer = tokenizer
            if not transforms:
                transforms = []
            self.__cls_global_edges = "cls_global_edges" in transforms

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

            kb_input_ids, kb_pool_inds = encode_kb_nodes(tokenizer, sample.kb)
            num_kb_nodes = max(kb_pool_inds) + 1 if kb_pool_inds else 0

            num_text_nodes = tokenized_text['input_ids'].shape[-1]
            num_text_subwords = num_text_nodes

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

            text_mask_inds, text_mask_values = build_identity_pool_mask(num_text_nodes)
            kb_mask_inds, kb_mask_values = build_average_pool_mask(kb_pool_inds)
            kb_mask_inds = [(i + num_text_nodes, j + num_text_subwords) for (i, j) in kb_mask_inds]

            mask_inds = text_mask_inds + kb_mask_inds
            mask_inds = [(0, *inds) for inds in mask_inds]

            mask_inds = torch.tensor(mask_inds, device=device).transpose(1, 0)
            mask_vals = torch.tensor(text_mask_values + kb_mask_values, device=device)
            total_pooled_nodes = num_text_nodes + num_kb_nodes
            node_mask = torch.sparse_coo_tensor(
                indices=mask_inds,
                values=mask_vals,
                size=(1, total_pooled_nodes, concat_ids.shape[-1]),
                is_coalesced=True,
                requires_grad=True,
                dtype=torch.float,
                device=device
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


            # Indices into a sparse array (batch, max_new_nodes, max_new_nodes, relation)
            # Need a 0 at the beginning for batch
            new_edges = []
            # The original token-to-token edges of a standard BERT model
            new_edges.extend((0, head, tail, TOKEN_TO_TOKEN_RELATION_ID) for (head, tail) in product(range(num_text_subwords), range(num_text_subwords)))
            # The KB edges, with indices adjusted
            orig_text_nodes = len(sample.target) + len(sample.context)
            max_node_index = orig_text_nodes + num_kb_nodes
            for e in sample.edges:
                if orig_text_nodes <= e.head_node_index < max_node_index:
                    head_list = [num_text_subwords + (e.head_node_index - orig_text_nodes)]
                elif e.head_node_index in expand_list:
                    head_list = expand_list[e.head_node_index]
                else:
                    continue
                if orig_text_nodes <= e.tail_node_index < max_node_index:
                    tail_list = [num_text_subwords + (e.tail_node_index - orig_text_nodes)]
                elif e.tail_node_index in expand_list:
                    tail_list = expand_list[e.tail_node_index]
                else:
                    continue
                new_edges.extend((0, head, tail, e.relation_id) for (head, tail) in product(head_list, tail_list))

            # Additional edges linking the CLS node to the external nodes
            if self.__cls_global_edges:
                global_edges = []
                for target in range(num_text_subwords, max_node_index):
                    global_edges.append((0,      0, target, TOKEN_TO_KB_RELATION_ID))
                    global_edges.append((0, target,      0, TOKEN_TO_KB_RELATION_ID))
                new_edges.extend(global_edges)


            new_edges.sort()
            new_edges = torch.tensor(new_edges, device=device).transpose(1, 0)

            return {
                "input_ids" : concat_ids,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "pooling_mask" : node_mask,
                "edge_indices": new_edges,
                "stance": torch.tensor(sample.stance.value, device=device)
            }

        def collate(self, samples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {
                "stance": torch.stack([s['stance'] for s in samples]),
                **collate_ids(self.__tokenizer, samples),
                **collate_graph_data(samples)
            }
    

class ConcatClassifier(StanceClassifier):
    def __init__(self, config: GatbertConfig):
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
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer
            self.__graph_encoder = ExternalClassifier.Encoder(tokenizer)
            self.__text_encoder = TextClassifier.Encoder(tokenizer)
    
        def encode(self, sample: GraphSample):
            assert isinstance(sample, GraphSample)
            encoded_graph = self.__graph_encoder.encode(sample)
            encoded_text = self.__text_encoder.encode(sample.to_sample())
            encoded_graph.pop('stance')
            return {
                "text": encoded_text,
                "graph": encoded_graph,
                'stance': encoded_text.pop('stance')
            }
    
        def collate(self, samples: List[Dict[str, TensorDict]]) -> TensorDict:
            rdict = {}
            rdict["stance"] = torch.stack([s['stance'] for s in samples])
    
            graph_samples = [s['graph'] for s in samples]
            rdict['graph'] = {
                **collate_ids(self.__tokenizer, graph_samples),
                **collate_graph_data(graph_samples, return_node_counts=True)
            }
            rdict['text'] = collate_ids(self.__tokenizer, [s['text'] for s in samples], return_attention_mask=True)
            return rdict


class TextClassifier(StanceClassifier):
    def __init__(self, config: GatbertConfig):
        super().__init__(config)
        self.bert = BertModel(config.wrapped)
        self.projection = torch.nn.Linear(
            self.bert.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )

    def load_pretrained_weights(self):
        # TODO: Find a cleaner way than just reinstantiating the object
        self.bert = BertModel.from_pretrained(self.config.base_model)

    def forward(self, *args, **kwargs):
        bert_output = self.bert(*args, **kwargs)
        last_hidden_state = bert_output['last_hidden_state'][:, 0]
        logits = self.projection(last_hidden_state)
        return logits

    class Encoder(Encoder):
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer

        def encode(self, sample: Sample | PretokenizedSample):
            if isinstance(sample, Sample):
                result = self.__tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=False, return_tensors='pt')
            elif isinstance(sample, PretokenizedSample):
                result = self.__tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=True, return_tensors='pt')
            else:
                raise ValueError(f"Invalid sample type {type(sample)}")
            result['stance'] = torch.tensor(sample.stance.value, device=result['input_ids'].device)
            return result

        def collate(self, samples: List[TensorDict]) -> TensorDict:
            return {**collate_ids(self.__tokenizer, samples, return_attention_mask=True), **collate_stance(samples)}
