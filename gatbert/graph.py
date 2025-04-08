# STL
import os
import csv
from typing import List, Tuple, Dict
from collections import defaultdict
# 3rd Party
import torch
import numpy as np
# Local
from .utils import open_gzip_or_plain
from .constants import SpecialRelation

EdgeList = List[Tuple[int, int, int]]
NodeList = List[int]

def get_entity_embeddings_path(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'entities.pkl')
def get_relation_embeddings_path(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'relations.pkl')
def get_entities_path(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, "entity_to_id.tsv.gz")
def get_relations_path(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, "relation_to_id.tsv.gz")
def get_triples_path(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'numeric_triples.tsv.gz')
def get_bert_triples_path(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'bert_triples.tsv.gz')

class GraphPaths:
    def __init__(self, graph_root: os.PathLike):
        """
        Path to raw ConceptNet assertions file.
        """

        self.assertions_path = os.path.join(graph_root, 'assertions.tsv')
        self.seeds_path = os.path.join(graph_root, "seeds.tsv")

        self.entity_embeddings_path = os.path.join(graph_root, 'entities.pkl')
        self.relation_embeddings_path = os.path.join(graph_root, 'relations.pkl')
        self.entities_path = os.path.join(graph_root, "entity_to_id.tsv.gz")
        self.relations_path = os.path.join(graph_root, "relation_to_id.tsv.gz")
        self.triples_path = os.path.join(graph_root, 'numeric_triples.tsv.gz')
        self.bert_triples_path = os.path.join(graph_root, 'bert_triples.tsv.gz')

type AdjMat = Dict[int, List[Tuple[int, int]]]

def read_bert_adj_mat(triples_path: os.PathLike, sim_threshold: float = 0.5):
    with open_gzip_or_plain(triples_path) as r:
        reader = csv.DictReader(r, delimiter='\t')
        reader = list(reader)
        triples = [
            (int(row['head']), int(row['tail']), int(row['relation']))
            for row in reader
            if float(row['similarity']) >= sim_threshold
        ]
    adj = defaultdict(list)
    for (head, tail, rel) in triples:
        adj[head].append((tail, rel))
    adj = dict(adj)
    return adj

def read_adj_mat(triples_path: os.PathLike, make_inverse_rels=True) -> AdjMat:
    heads = []
    tails = []
    rels = []
    with open_gzip_or_plain(triples_path) as r:
        reader = csv.DictReader(r, delimiter='\t')
        reader = list(reader)
        for row in reader:
            heads.append(int(row['head']))
            tails.append(int(row['tail']))
            rels.append(int(row['relation']))
    heads = np.array(heads)
    tails = np.array(tails)
    rels = np.array(rels)

    adj = defaultdict(list)
    if make_inverse_rels:
        rels = 2 * rels
        inv_rels = rels + 1
        for (head, tail, rel, inv_rel) in zip(heads, tails, rels, inv_rels):
            adj[head].append((tail, rel))
            adj[tail].append((head, inv_rel))
    else:
        for (head, tail, rel) in zip(heads, tails, rels):
            adj[head].append((tail, rel))

    adj = dict(adj) # If you don't intend any more writes, you should convert your defaultdict to a dict
    return adj

def update_adj_mat(sink: AdjMat, source: AdjMat) -> AdjMat:
    for (head, edges) in source.items():
        if head not in sink:
            sink[head] = []
        sink[head].extend(edges)
    return sink

def read_entitites(p: os.PathLike):
    uri2id = {}
    with open_gzip_or_plain(p) as r:
        reader = csv.DictReader(r, delimiter='\t')
        for row in reader:
            uri2id[row['label']] = int(row['id'])
    return uri2id

def read_relations(p: os.PathLike):
    rel2id = {}
    with open_gzip_or_plain(p) as r:
        reader = csv.DictReader(r, delimiter='\t')
        for row in reader:
            label = row['label']
            forward_id = int(row['id'])
            rel2id[label] = forward_id
            rel2id[f'{label}/inv'] = forward_id + 1
    return rel2id

def get_n_relations(graph_path: os.PathLike):
    return len(read_relations(get_relations_path(graph_path)))

def load_kb_embeddings(graph_path: os.PathLike, pretrained_relations=False) -> Tuple[torch.Tensor, torch.Tensor]:
    entity_embeddings =  torch.load(get_entity_embeddings_path(graph_path), weights_only=False)
    rel_path = get_relation_embeddings_path(graph_path)

    expected_relations = get_n_relations(graph_path)
    total_relations = expected_relations + len(SpecialRelation)
    rel_embeddings = torch.nn.Embedding(total_relations, entity_embeddings.weight.shape[1])
    # TODO: Should I initialize the special relations with 0's like I do in EdgeEmbeddings ...?
    if pretrained_relations and os.path.exists(rel_path):
        embedding_obj = torch.load(rel_path, weights_only=False)
        pretrained_embeds = embedding_obj.weight.data
        assert pretrained_embeds.shape == (expected_relations, entity_embeddings.weight.shape[1])
        rel_embeddings.weight.data[:expected_relations] = pretrained_embeds
    return entity_embeddings, rel_embeddings
