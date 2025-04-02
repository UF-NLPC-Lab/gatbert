# STL
from typing import List
from itertools import product
import os
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast
from transformers.models.bert.modeling_bert import BertConfig
# Local
from .base_module import StanceModule
from .gatbert import GatbertEncoder, GatbertEmbeddings
from .constants import Stance, DEFAULT_MODEL, MAX_KB_NODES, SpecialRelation
from .config import GatbertConfig
from .encoder import *
from .graph import *


class HybridModule(StanceModule):
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
        self.gatbert = GatbertEncoder(self.config)
        self.gatbert.load_pretrained_weights(pretrained_model_obj.encoder)
        self.projection = torch.nn.Linear(
            self.config.hidden_size,
            out_features=len(Stance),
            bias=False
        )
        self.__preprocessor = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), graph)

    @property
    def encoder(self) -> Encoder:
        return self.__preprocessor

    def forward(self, input_ids, kb_mask, edge_indices, position_ids = None, token_type_ids = None):
        node_embeddings = self.embeddings(input_ids=input_ids,
                                     kb_mask=kb_mask,
                                     position_ids=position_ids,
                                     token_type_ids=token_type_ids)
        # TODO: incorporate relational embeddings ?
        final_node_states, _ = self.gatbert(node_embeddings, edge_indices=edge_indices)
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
    