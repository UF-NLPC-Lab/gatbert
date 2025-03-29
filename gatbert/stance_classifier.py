import abc
from typing import List, Literal
from itertools import product
import os
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertConfig
# Local
from .gatbert import GatbertEncoder, GatbertEmbeddings
from .constants import Stance, DEFAULT_MODEL, MAX_KB_NODES, SpecialRelation
from .config import GatbertConfig
from .encoder import *
from .graph import *
from .cgcn import Cgcn


class StanceClassifier(torch.nn.Module):

    @abc.abstractmethod
    def get_encoder(self) -> Encoder:
        pass

    def get_grads(self):
        return []


class HybridClassifier(StanceClassifier):
    def __init__(self,
                graph: os.PathLike,
                pretrained_model: str = DEFAULT_MODEL,
                ):
        super().__init__()

        self.config = GatbertConfig(
            BertConfig.from_pretrained(pretrained_model),
            n_relations=get_n_relations(graph),
        )

        pretrained_model_obj = BertModel.from_pretrained(pretrained_model)
        self.embeddings = GatbertEmbeddings(self.config, graph)
        self.embeddings.load_pretrained_weights(pretrained_model_obj.embeddings)
        self.encoder = GatbertEncoder(self.config)
        self.encoder.load_pretrained_weights(pretrained_model_obj.encoder)
        self.projection = torch.nn.Linear(
            self.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
        self.__preprocessor = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), graph)

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
                     graph: os.PathLike):
            self.__tokenizer = tokenizer

            self.__uri2id = read_entitites(get_entities_path(graph))
            self.__total_relations = get_n_relations(graph) + len(SpecialRelation)

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

            kb_input_ids = torch.tensor([[self.__uri2id[node] for node in sample.kb[:MAX_KB_NODES]]], dtype=torch.int64)
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
            new_edges.extend((0, head, tail, SpecialRelation.TOKEN_TO_TOKEN.value) for (head, tail) in product(range(new_text_nodes), range(new_text_nodes)))
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

            # Ensure no negative rel indices
            new_edges = [(*others, rel % self.__total_relations) for (*others, rel) in new_edges]
            # When we use these in sparse_coo arrays later, they'll need to be sorted
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
                 num_graph_layers: int = 2,
                 pretrained_relations: bool = False):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)

        self.entity_embeddings, self.relation_embeddings = load_kb_embeddings(graph, pretrained_relations)
        (_, self.entity_embed_dim) = self.entity_embeddings.weight.shape
        (self.n_relations, self.relation_embed_dim) = self.relation_embeddings.weight.shape

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
        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), graph)
    
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

    def get_grads(self):
        return []
        weight = self.pred_head.weight
        grad = weight.grad
        split_index = 2 * self.bert.config.hidden_size
        with torch.no_grad():
            return [
                ("z_text_weight_norm", torch.linalg.norm(weight[:, :split_index])),
                ("z_graph_weight_norm", torch.linalg.norm(weight[:, split_index:])),
                ("z_text_weight_grad_norm", torch.linalg.norm(grad[:, :split_index])),
                ("z_graph_weight_grad_norm", torch.linalg.norm(grad[:, split_index:]))
            ]

