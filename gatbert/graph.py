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

def get_entity_embeddings(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'entities.pkl')
def get_relation_embeddings(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'relations.pkl')
def get_triples(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'numeric_triples.tsv.gz')

@dataclasses.dataclass
class CNGraph:

    uri2id: Dict[str, int]
    id2uri: Dict[int, str]
    tok2id: Dict[str, int]
    rel2id: Dict[str, int]
    adj: Dict[int, List[Tuple[int, int]]]

    def __init__(self, uri2id, id2uri, adj, rel2id):
        self.uri2id = uri2id
        self.id2uri = id2uri
        self.adj = adj
        self.rel2id = rel2id

        matches = map(lambda pair: (CN_URI_PATT.fullmatch(pair[0]), pair[1]), self.uri2id.items())
        matches = filter(lambda pair: pair[0], matches)
        self.tok2id = {match.group(1):id for (match, id) in matches}


    def __repr__(self):
        return "<CNGraph>"

    @staticmethod
    def read(path: str):
        assert os.path.isdir(path)
        return CNGraph.from_pykeen(path)

    @staticmethod
    def read_entitites(dir_or_path: os.PathLike):
        dir_or_path = pathlib.Path(dir_or_path)
        entity_path = dir_or_path.joinpath("entity_to_id.tsv") if os.path.isdir(dir_or_path) else dir_or_path
        uri2id = {}
        with open_gzip_or_plain(entity_path) as r:
            reader = csv.DictReader(r, delimiter='\t')
            for row in reader:
                uri2id[row['label']] = int(row['id'])
        return uri2id

    @staticmethod
    def read_relations(pykeen_dir: os.PathLike):
        rel2id = {}
        with open_gzip_or_plain(pathlib.Path(pykeen_dir).joinpath("relation_to_id.tsv")) as r:
            reader = csv.DictReader(r, delimiter='\t')
            for row in reader:
                label = row['label']
                forward_id = int(row['id'])
                rel2id[label] = forward_id
                rel2id[f'{label}/inv'] = forward_id + 1
        return rel2id

    @staticmethod
    def from_pykeen(pykeen_dir: str):
        pykeen_dir  = pathlib.Path(pykeen_dir)

        uri2id = CNGraph.read_entitites(pykeen_dir)
        id2uri = {v:k for (k, v) in uri2id.items()}

        # TODO: don't use pandas for this
        edge_df = pd.read_csv(pykeen_dir.joinpath('numeric_triples.tsv.gz'), compression='gzip', delimiter='\t')
        heads = edge_df['head'].apply(int) # head is also a DF method. Use [] to circumvent that
        tails = edge_df['tail'].apply(int)  # Same for tail
        rels = 2 * edge_df.relation.apply(int)
        inv_rels = rels + 1
        adj = defaultdict(list)
        for (head, tail, rel, inv_rel) in zip(heads, tails, rels, inv_rels):
            adj[head].append((tail, rel))
            adj[tail].append((head, inv_rel))
        adj = dict(adj) # If you don't intend any more writes, you should convert your defaultdict to a dict

        rel2id = CNGraph.read_relations(pykeen_dir)

        return CNGraph(
            uri2id=uri2id,
            id2uri=id2uri,
            adj=adj,
            rel2id=rel2id
        )