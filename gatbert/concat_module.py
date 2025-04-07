# STL
import os
from typing import List, Dict
# 3rd Party
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast
from pykeen.models.unimodal.compgcn import CompGCN
from pykeen.nn.representation import SingleCompGCNRepresentation
from pykeen.triples.triples_factory import TriplesFactory
# Local
from .sample import Sample
from .base_module import StanceModule
from .constants import DEFAULT_MODEL, Stance
from .encoder import Encoder, collate_ids, keyed_pad, keyed_scalar_stack, encode_text
from .types import TensorDict

class KBAttention(torch.nn.Module):
    def __init__(self,
                 text_dim: int,
                 kb_dim: int):
        super().__init__()

        self.q_proj = torch.nn.Linear(text_dim, kb_dim, bias=False)
        self.k_proj = torch.nn.Linear(kb_dim, kb_dim, bias=False)
        self.v_proj = torch.nn.Linear(kb_dim, kb_dim, bias=False)
        self.out_proj = torch.nn.Linear(kb_dim, kb_dim, bias=False)
        self.text_dim = text_dim
        self.kb_dim = kb_dim
        self.register_buffer("dp_scale", torch.sqrt(torch.tensor(self.kb_dim)))

    def forward(self, text: torch.Tensor, kb: torch.Tensor):
        Q = self.q_proj(text)
        K = self.k_proj(kb)

        Q_scaled = Q / self.dp_scale
        act = torch.matmul(Q_scaled, K.transpose(-1, -2))
        weights = torch.softmax(act, dim=-1)

        V = self.v_proj(kb)
        att_vals = torch.matmul(weights, V)

        # TODO: add dropout?

        return att_vals

class ConcatModule(StanceModule):

    def __init__(self,
                 graph_dir: os.PathLike,
                 pretrained_model: str = DEFAULT_MODEL,
                 joint_loss: bool = False,
                 num_graph_layers: int = 2,
                 node_embed_dim: int = 64,

                 # seed mask file
                 ):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__()
        self.save_hyperparameters()

        triples_factory = TriplesFactory.from_path_binary(graph_dir)

        cgcn = CompGCN(
            embedding_dim=node_embed_dim,
            triples_factory=triples_factory,
        )
        self.cgcn: SingleCompGCNRepresentation = cgcn.entity_representations[0]

        self.bert = BertModel.from_pretrained(pretrained_model)
        self.context_att = KBAttention(self.bert.config.hidden_size, node_embed_dim)
        self.target_att = KBAttention(self.bert.config.hidden_size, node_embed_dim)

        self.graph_head = torch.nn.Linear(2 * node_embed_dim,        len(Stance), bias=False)
        self.text_head  = torch.nn.Linear(2 * self.bert.config.hidden_size, len(Stance), bias=False)
        self.__encoder = self.Encoder(BertTokenizerFast.from_pretrained(pretrained_model))

        # Load a seed mask

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
                context_text_mask):
        # Force a new forward pass of the graph
        self.cgcn.combined.enriched_representations = None

        # (1) Encode text
        bert_out = self.bert(**text)
        hidden_states = bert_out.last_hidden_state
        target_text_vec = self.masked_average(target_text_mask, hidden_states)
        context_text_vec = self.masked_average(context_text_mask, hidden_states)

        # FIXME: Use seed mask to split these into H_target and H_context
        graph_encodings = self.cgcn()

        target_node_vec = self.target_att(target_text_vec, graph_encodings)
        context_node_vec = self.context_att(context_text_vec, graph_encodings)

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
        def __init__(self, tokenizer: PreTrainedTokenizerFast):
            self.__tokenizer = tokenizer

        def encode(self, sample: Sample):
            assert isinstance(sample, Sample)
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

            return {
                "text": text_encoding,
                "target_text_mask": target_text_mask,
                "context_text_mask": context_text_mask,
                "stance": torch.tensor([sample.stance.value]),
            }
    
        def collate(self, samples: List[Dict[str, TensorDict]]) -> TensorDict:
            rdict = {}

            rdict['text'] = collate_ids(self.__tokenizer, [s['text'] for s in samples], return_attention_mask=True)
            rdict['target_text_mask'] = keyed_pad(samples, 'target_text_mask')
            rdict['context_text_mask'] = keyed_pad(samples, 'context_text_mask')


            rdict["stance"] = keyed_scalar_stack(samples, 'stance')
    
            return rdict
