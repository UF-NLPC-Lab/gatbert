# STL
import os
from typing import List, Dict
import logging
import csv
import time
# 3rd Party
from tqdm import tqdm
import pykeen
import torch
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast
from pykeen.models.unimodal.compgcn import CompGCN
from pykeen.nn.representation import SingleCompGCNRepresentation
from pykeen.triples.triples_factory import TriplesFactory, get_mapped_triples
import pandas as pd
# Local
from .sample import Sample
from .base_module import StanceModule
from .constants import DEFAULT_MODEL, Stance
from .encoder import Encoder, collate_ids, keyed_pad, keyed_scalar_stack, encode_text, get_text_masks
from .types import TensorDict
from .graph import GraphPaths
from .utils import time_block
from .bert_module import BertModule

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
                 use_bert_triples: bool = False
                 # seed mask file
                 ):
        """
        Args:
            pretrained_model_name: model to load for text portion of the model
            config: config for the graph portion of the model
        """
        super().__init__()
        self.save_hyperparameters()

        graph_paths = GraphPaths(graph_dir)
        triples_factory = TriplesFactory.from_path_binary(graph_dir)
        if use_bert_triples:
            df = pd.read_csv(graph_paths.bert_triples_path,
                             usecols=["head", "tail", "relation"],
                             dtype=int, sep='\t',
                             compression='gzip')
            unique_edges = set()
            for _, row in tqdm(df.iterrows(), total=len(df)):
                head = row['head']
                tail = row['tail']
                if (tail, head) not in unique_edges:
                    unique_edges.add((head, tail))
            e2id = dict()
            for i, (head, tail) in enumerate(tqdm(unique_edges)):
                e2id[head, tail] = i
                e2id[tail, head] = i
            df['edge_id'] = df.apply(lambda row: e2id[row['head'], row['tail']], axis=1)
            unique_df = df.drop_duplicates(subset='edge_id')
            rel_mapping = pd.read_csv(graph_paths.relations_path, sep='\t', compression='gzip')
            relatedto_id = rel_mapping[rel_mapping.label.apply(lambda x: x.lower().replace("/r/", "")) == "relatedto"].iloc[0].id
            bert_triples = unique_df[["head", "relation", "tail"]].to_numpy()
            bert_triples[:, 1] = relatedto_id
            bert_triples = torch.tensor(bert_triples)
            triples_factory.clone_and_exchange_triples(
                mapped_triples=torch.concatenate([
                    bert_triples,
                    get_mapped_triples(factory=triples_factory)
                ])
            )
        target_inds_l = []
        context_inds_l = []
        ent2id = triples_factory.entity_to_id
        with open(graph_paths.seeds_path, 'r') as r:
            for (seed, in_target, in_context) in csv.reader(r, delimiter='\t'):
                if seed not in ent2id:
                    continue # Pykeen discards an entity if there's no edge for it
                in_target = int(in_target)
                in_context = int(in_context)
                (index,) = triples_factory.entities_to_ids([seed])
                if in_target:
                    target_inds_l.append(index)
                if in_context:
                    context_inds_l.append(index)
        target_inds_l = sorted(target_inds_l)
        context_inds_l = sorted(context_inds_l)

        self.register_buffer("target_inds", torch.tensor(target_inds_l, dtype=torch.long))
        self.register_buffer("context_inds", torch.tensor(context_inds_l, dtype=torch.long))

        self.bert = BertModule(pretrained_model)
        bert_config = self.bert.bert.config

        cgcn = CompGCN(
            embedding_dim=node_embed_dim,
            triples_factory=triples_factory,
            encoder_kwargs=dict(
                num_layers=num_graph_layers,
                layer_kwargs=dict(composition="SubtractionCompositionModule")
            ),
            random_seed=pykeen.utils.NoRandomSeedNecessary # Random seed already set by lightning
        )
        self.cgcn: SingleCompGCNRepresentation = cgcn.entity_representations[0]
        if os.path.exists(graph_paths.entity_embeddings_path):
            combined_cgcn = self.cgcn.combined
            save_ent_embeddings = torch.load(graph_paths.entity_embeddings_path, weights_only=False)
            combined_cgcn.entity_representations._embeddings.load_state_dict(save_ent_embeddings.state_dict())
            if os.path.exists(graph_paths.relation_embeddings_path):
                rel_ent_embeddings = torch.load(graph_paths.relation_embeddings_path, weights_only=False)
                combined_cgcn.relation_representations._embeddings.load_state_dict(rel_ent_embeddings.state_dict())
            else:
                logging.warning("Loaded entity embeddings from %s but could not find relation embeddings", graph_dir)
        else:
                logging.warning("Could not find entity embeddings in %s.", graph_dir)
        self.context_att = KBAttention(bert_config.hidden_size, node_embed_dim)
        self.target_att = KBAttention(bert_config.hidden_size, node_embed_dim)

        hidden_size = 283 # Alloway's hparam
        self.ff = torch.nn.Sequential(
            torch.nn.Dropout(p=0.20463604390811982),
            torch.nn.Linear(2 * bert_config.hidden_size + 2 * node_embed_dim, hidden_size, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, len(Stance), bias=True)
        )


        self.__encoder = BertModule.Encoder(BertTokenizerFast.from_pretrained(pretrained_model), max_context_length=200, max_target_length=5)


    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    def training_step(self, batch, batch_idx):
        # Force a new forward pass of the graph
        self.cgcn.combined.enriched_representations = None
        labels = batch.pop("stance")
        # Calls the forward method defined in subclass
        logits = self(**batch)

        joint_loss = torch.nn.functional.cross_entropy(logits, labels)
        total_loss = joint_loss

        self.log("loss", total_loss)
        return total_loss

    # If we were training right before starting one of these,
    # we don't want the backward calculations for these reprs
    # hanging around in memory. So force recomputation once
    # at the beginning of the epoch
    def on_validation_epoch_start(self):
        self.cgcn.combined.enriched_representations = None
    def on_test_epoch_start(self):
        self.cgcn.combined.enriched_representations = None

    def forward(self, **kwargs):

        # (1) Encode text
        _, context_text_vec, target_text_vec = self.bert(**kwargs)

        graph_encodings = self.cgcn()

        target_seed_encodings = graph_encodings[self.target_inds]
        target_node_vec = self.target_att(target_text_vec, target_seed_encodings)

        context_seed_encodings = graph_encodings[self.context_inds]
        context_node_vec = self.context_att(context_text_vec, context_seed_encodings)

        feature_vec = torch.concatenate([context_text_vec, target_text_vec, context_node_vec, target_node_vec], dim=-1)
        logits = self.ff(feature_vec)
        return logits
