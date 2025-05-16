import pathlib
# 3rd Party
import numpy as np
import torch
from torch.utils.data import random_split
import torch_geometric
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import lightning as L
from lightning.pytorch.loggers import CSVLogger
# Local
from .cn import CN
from .utils import batched

class RGCN(torch.nn.Module):
    def __init__(self,
                 n_entities: int,
                 n_relations: int,
                 dim: int,
                 num_bases: int = 4,
                 dropout: float = 0.25):
        super().__init__()
        self.entity_embed = torch.nn.Embedding(n_entities, dim)

        self.conv1 = RGCNConv(dim, dim, n_relations, num_bases=num_bases)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = RGCNConv(dim, dim, n_relations, num_bases=num_bases)

        self.relation_embed = torch.nn.Embedding(n_relations, dim)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, entity, edge_index, edge_type, triplets=None, labels=None):
        node_state = self.entity_embed(entity)
        node_state = self.conv1(node_state, edge_index, edge_type)
        node_state = self.relu(node_state)
        node_state = self.dropout(node_state)
        node_state = self.conv2(node_state, edge_index, edge_type)

        loss = None
        if triplets is not None and labels is not None:
            head_state = node_state[triplets[:, 0]]
            rel_embedding = self.relation_embed(triplets[:, 1])
            tail_state = node_state[triplets[:, 2]]
            logits = torch.sum(head_state * rel_embedding * tail_state, dim=-1)
            loss = self.loss_fn(logits, labels)
        return node_state, loss

class CNTrainModule(L.LightningModule):
    def __init__(self,
                 assertions_path: pathlib.Path,
                 dim: int = 100,
                 pos_triples: int = 50000,
                 neg_ratio: int = 1,
                 message_ratio: float = 0.5):
        self.cn = CN(assertions_path)
        self.rgcn = RGCN(len(self.cn.node2id), len(self.cn.relation2id), dim=dim)
        self.cn_edges = np.array(self.cn.edges)
        self.pos_triples = pos_triples
        self.neg_ratio = neg_ratio
        self.message_ratio = message_ratio

    def train_dataloader(self):
        shuffled_edges = np.random.permutation(self.cn_edges)

        for raw_batch in batched(shuffled_edges, self.pos_triples):
            raw_batch = np.stack(raw_batch, axis=-1)
            head, rel, tail = raw_batch
            unique_nodes, new_edges = np.unique((head, tail), return_inverse=True)

            new_edges = np.stack([new_edges[0], rel, new_edges[1]], axis=0)

            n_message_edges = int(self.message_ratio * new_edges.shape[-1])
            split_ids = np.random.choice(np.arange(new_edges.shape[1]), size=n_message_edges, replace=False)
            message_edges = new_edges[split_ids]
            edge_index = np.stack([message_edges[0], message_edges[2]], axis=0)
            edge_attr = message_edges[1]

            # Make triples to be scored
            neg_samples = np.tile(new_edges, (self.neg_ratio, 1))
            alterations = np.random.choice(unique_nodes.shape[0], size=neg_samples.shape[0])
            alter_head = np.random.uniform(size=neg_samples.shape[0]) > 0.5
            neg_samples[alter_head, 0] = alterations[alter_head]
            neg_samples[~alter_head, 2] = alterations[~alter_head]
            triples = np.concatenate([new_edges, neg_samples], axis=0)
            y = np.concatenate([np.ones(new_edges.shape[0]), np.zeros(neg_samples.shape[0])])

            d = Data(x=torch.tensor(unique_nodes),
                     edge_index=torch.tensor(edge_index),
                     edge_attr=torch.tensor(edge_attr),
                     y=torch.tensor(y))
            # Non-standard attributes
            d.triples = torch.tensor(triples)
            pass

        return super().train_dataloader()

if __name__ == "__main__":
    assertions_path = "./temp/filter_graph.tsv"
    out_dir = "graph_logs/"
    mod = CNTrainModule(assertions_path)
    logger = CSVLogger(save_dir=out_dir, name=None)

    trainer = L.Trainer(
        max_epochs=300,
        logger=logger,
        deterministic=True,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(model=mod, train_dataloaders=mod)
    pass