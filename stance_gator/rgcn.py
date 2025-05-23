import os
import pathlib
from typing import Iterable
# 3rd Party
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import RGCNConv
import torch_geometric.data
import torch_geometric.loader
import lightning as L
import torch_scatter
# Local
from .sample import Sample
from .cn import CN
from .utils import batched
from .data import SPACY_PIPES, extract_lemmas

class RGCN(torch.nn.Module):
    def __init__(self,
                 n_entities: int,
                 n_relations: int,
                 dim: int,
                 num_bases: int = 4,
                 dropout: float = 0.25):
        super().__init__()
        self.dim = dim


        self.entity_embed = torch.nn.Embedding(n_entities, dim)

        self.conv1 = RGCNConv(dim, dim, n_relations, num_bases=num_bases)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = RGCNConv(dim, dim, n_relations, num_bases=num_bases)

        self.relation_embed = torch.nn.Embedding(n_relations, dim)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index, edge_type, triples=None, y=None):
        node_state = self.entity_embed(x)
        node_state = self.conv1(node_state, edge_index, edge_type)
        node_state = self.relu(node_state)
        node_state = self.dropout(node_state)
        node_state = self.conv2(node_state, edge_index, edge_type)

        loss = None
        if triples is not None and y is not None:
            head_state = node_state[triples[0]]
            rel_embedding = self.relation_embed(triples[1])
            tail_state = node_state[triples[2]]
            logits = torch.sum(head_state * rel_embedding * tail_state, dim=-1)
            loss = self.loss_fn(logits, y)
        return node_state, loss


def text2edges(cn: CN, s: Sample) -> np.ndarray:
    node2id = cn.node2id
    adj = cn.adj
    rev_adj = cn.rev_adj
    if s.is_split_into_words:
        text = " ".join(s.target) + " " + " ".join(s.context)
    else:
        text = s.target + ' ' + s.context

    lang = s.lang or 'en'
    pipeline = SPACY_PIPES[lang]
    lemmas = [lemma for lemma in extract_lemmas(pipeline, text) if lemma in node2id]
    lemma_ids = [node2id[lemma] for lemma in lemmas]

    forward_edges = [(head, rel, tail) for head in lemma_ids for (rel, tail) in adj.get(head, [])]
    rev_edges = [(head, rel, tail) for tail in lemma_ids for (rel, head) in rev_adj.get(tail, [])]
    edges = np.unique(forward_edges + rev_edges, axis=0).transpose()
    return edges


def make_graph_sample(raw_batch: np.ndarray,
                      n_relations,
                      message_ratio: float = 1.0,
                      with_triples: bool = False,
                      neg_ratio: int = 1
                      ) -> torch_geometric.data.Data:
    if raw_batch.size == 0:
        return torch_geometric.data.Data(
            x=torch.empty(0, dtype=torch.long),
            edge_index=torch.empty([2, 0], dtype=torch.long),
            edge_attr=torch.empty(0, dtype=torch.long)
        )

    head, rel, tail = raw_batch
    unique_nodes, head_tail_inds = np.unique((head, tail), return_inverse=True)

    relabeled_edges = np.stack([head_tail_inds[0], rel, head_tail_inds[1]], axis=0)

    # Make triples to be scored
    if with_triples:
        neg_samples = np.tile(relabeled_edges, (1, neg_ratio))
        alterations = np.random.choice(unique_nodes.shape[0], size=neg_samples.shape[1])
        alter_head = np.random.uniform(size=neg_samples.shape[1]) > 0.5
        neg_samples[0, alter_head] = alterations[alter_head]
        neg_samples[2, ~alter_head] = alterations[~alter_head]
        triples = np.concatenate([relabeled_edges, neg_samples], axis=1)
        y = np.concatenate([np.ones(relabeled_edges.shape[1]), np.zeros(neg_samples.shape[1])])
        triples = torch.tensor(triples)
        y = torch.tensor(y)
    else:
        triples = None
        y = None

    if message_ratio < 1:
        n_message_edges = int(message_ratio * relabeled_edges.shape[-1])
        split_ids = np.random.choice(np.arange(relabeled_edges.shape[1]), size=n_message_edges, replace=False)
        message_edges = relabeled_edges[:, split_ids]
    else:
        message_edges = relabeled_edges
    # Only do reversal on edges used for message passing
    message_edges = CNEncoder.add_reverse_edges(message_edges, n_relations)
    edge_index = np.stack([message_edges[0], message_edges[2]], axis=0)
    edge_attr = message_edges[1]
    # Don't need to add self-loops; RGCNConv class already does that by default

    d = torch_geometric.data.Data(x=torch.tensor(unique_nodes),
             edge_index=torch.tensor(edge_index),
             edge_attr=torch.tensor(edge_attr),
             y=y)
    # Non-standard attributes
    d.triples = triples
    return d

class CNEncoder(L.LightningModule):
    def __init__(self,
                 cn: pathlib.Path,
                 dim: int = 256,
                 pos_triples: int = 50000,
                 neg_ratio: int = 1,
                 message_ratio: float = 0.5,
                 reg_weight: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.reg_weight = reg_weight
        self.cn = CN(cn)
        self.rgcn = RGCN(
            len(self.cn.node2id),
            len(self.cn.relation2id) * 2, # Double the number of relations to include inverse ones
            dim=dim)
        self.cn_triples = np.array(self.cn.triples)


        # Only relevant to training, not inference
        self.pos_triples = pos_triples
        self.neg_ratio = neg_ratio
        self.message_ratio = message_ratio

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
        
    def training_step(self, batch: torch_geometric.data.Data, batch_idx):
        _, loss_ce = self.rgcn(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_attr,
            triples=batch.triples,
            y=batch.y
        )

        reg = torch.mean(torch.pow(self.rgcn.entity_embed.weight, 2)) + \
            torch.mean(torch.pow(self.rgcn.relation_embed.weight, 2))
        loss = loss_ce + self.reg_weight * reg
        batch_size = batch.x.shape[0]
        self.log('loss/ce', loss_ce, on_epoch=True, on_step=False, batch_size=batch_size)
        self.log('loss/reg', reg, on_epoch=True, on_step=False, batch_size=batch_size)
        self.log('loss', loss, on_epoch=True, on_step=False, batch_size=batch_size)

        return loss

    def predict_step(self, batch: torch_geometric.data.Data, batch_idx):
        return self(batch)

    def forward(self, batch):
        node_states, _ = self.rgcn(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_attr
        )
        mean_node_state = torch_scatter.scatter(src=node_states,
                                                index=batch.batch,
                                                dim=0,
                                                dim_size=batch.ptr.shape[0] - 1,
                                                reduce='mean')

        return mean_node_state

    @staticmethod
    def add_reverse_edges(edges: np.ndarray, n_rels: int) -> np.ndarray:
        rev = np.stack([edges[2], edges[1] + n_rels, edges[0]])
        return np.concatenate([edges, rev], axis=1)

    def __make_sample(self, raw_batch: np.ndarray, with_triples=False) -> torch_geometric.data.Data:
        if raw_batch.size == 0:
            return torch_geometric.data.Data(
                x=torch.empty(0, dtype=torch.long),
                edge_index=torch.empty([2, 0], dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long)
            )

        head, rel, tail = raw_batch
        unique_nodes, head_tail_inds = np.unique((head, tail), return_inverse=True)

        relabeled_edges = np.stack([head_tail_inds[0], rel, head_tail_inds[1]], axis=0)

        # Make triples to be scored
        if with_triples:
            neg_samples = np.tile(relabeled_edges, (1, self.neg_ratio))
            alterations = np.random.choice(unique_nodes.shape[0], size=neg_samples.shape[1])
            alter_head = np.random.uniform(size=neg_samples.shape[1]) > 0.5
            neg_samples[0, alter_head] = alterations[alter_head]
            neg_samples[2, ~alter_head] = alterations[~alter_head]
            triples = np.concatenate([relabeled_edges, neg_samples], axis=1)
            y = np.concatenate([np.ones(relabeled_edges.shape[1]), np.zeros(neg_samples.shape[1])])
            triples = torch.tensor(triples)
            y = torch.tensor(y)
        else:
            triples = None
            y = None

        n_message_edges = int(self.message_ratio * relabeled_edges.shape[-1])
        split_ids = np.random.choice(np.arange(relabeled_edges.shape[1]), size=n_message_edges, replace=False)
        message_edges = relabeled_edges[:, split_ids]
        # Only do reversal on edges used for message passing
        message_edges = CNEncoder.add_reverse_edges(message_edges, len(self.cn.relation2id))
        edge_index = np.stack([message_edges[0], message_edges[2]], axis=0)
        edge_attr = message_edges[1]
        # Don't need to add self-loops; RGCNConv class already does that by default

        d = torch_geometric.data.Data(x=torch.tensor(unique_nodes),
                 edge_index=torch.tensor(edge_index),
                 edge_attr=torch.tensor(edge_attr),
                 y=y)
        # Non-standard attributes
        d.triples = triples
        return d

    def train_dataloader(self):
        shuffled_edges = np.random.permutation(self.cn_triples)
        samples = []
        for raw_batch in batched(shuffled_edges, self.pos_triples):
            raw_batch = np.stack(raw_batch, axis=-1)
            samples.append(self.__make_sample(raw_batch, with_triples=True))
        return torch_geometric.loader.DataLoader(samples)

    def make_predict_dataloader(self, samples: Iterable[Sample], batch_size=64):
        graph_samples = []
        node2id = self.cn.node2id
        adj = self.cn.adj
        rev_adj = self.cn.rev_adj
        samples = list(samples)
        for s in tqdm(samples, desc="Extracting subgraphs for samples"):
            # Luo et al. only used tokens from the context to extract the original subgraph,
            # but for embedding samples they use both target and context tokens
            if s.is_split_into_words:
                text = " ".join(s.target) + " " + " ".join(s.context)
            else:
                text = s.target + ' ' + s.context

            lang = s.lang or 'en'
            pipeline = SPACY_PIPES[lang]
            lemmas = [lemma for lemma in extract_lemmas(pipeline, text) if lemma in node2id]
            lemma_ids = [node2id[lemma] for lemma in lemmas]

            forward_edges = [(head, rel, tail) for head in lemma_ids for (rel, tail) in adj.get(head, [])]
            rev_edges = [(head, rel, tail) for tail in lemma_ids for (rel, head) in rev_adj.get(tail, [])]
            edges = np.unique(forward_edges + rev_edges, axis=0).transpose()
            graph_samples.append(self.__make_sample(edges, with_triples=False))
        return torch_geometric.loader.DataLoader(graph_samples, batch_size=batch_size)
