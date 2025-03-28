# STL
import os
import json
import gzip
import csv
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import dataclasses
import pathlib
# 3rd Party
import pandas as pd
# Local
from .constants import CN_URI_PATT
from .utils import open_gzip_or_plain

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

type AdjMat = Dict[int, List[Tuple[int, int]]]

def read_adj_mat(triples_path: os.PathLike, make_inverse_rels=True) -> AdjMat:
    # TODO: don't use pandas for this
    edge_df = pd.read_csv(triples_path, compression='gzip', delimiter='\t')
    heads = edge_df['head'].apply(int) # head is also a DF method. Use [] to circumvent that
    tails = edge_df['tail'].apply(int)  # Same for tail

    adj = defaultdict(list)
    if make_inverse_rels:
        rels = 2 * edge_df.relation.apply(int)
        inv_rels = rels + 1
        for (head, tail, rel, inv_rel) in zip(heads, tails, rels, inv_rels):
            adj[head].append((tail, rel))
            adj[tail].append((head, inv_rel))
    else:
        rels = edge_df.relation.apply(int)
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