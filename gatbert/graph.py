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

EdgeList = List[Tuple[int, int, int]]
NodeList = List[int]

def get_entity_embeddings(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'entities.pkl')
def get_relation_embeddings(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'relations.pkl')
def get_triples(graph_root: os.PathLike) -> os.PathLike:
    return os.path.join(graph_root, 'numeric_triples.tsv.gz')

def get_relation_mapping(graph_root: os.PathLike) -> Dict[str, int]:
    rdict = {}
    with gzip.open(os.path.join(graph_root, "relation_to_id.tsv.gz"), 'r') as r:
        reader = csv.DictReader(r)
        for row in reader:
            label = row['label']
            forward_id = int(row['id'])
            rdict[label] = forward_id
            rdict[f'{label}/inv'] = forward_id + 1
    return rdict

@dataclasses.dataclass
class CNGraph:
    uri2id: Dict[str, int]
    id2uri: Dict[int, str]
    tok2id: Dict[str, int]
    adj: Dict[int, List[Tuple[int, int]]]

    def __init__(self, uri2id, id2uri, adj):
        self.uri2id = uri2id
        self.id2uri = id2uri
        self.adj = adj

        matches = map(lambda pair: (CN_URI_PATT.fullmatch(pair[0]), pair[1]), self.uri2id.items())
        matches = filter(lambda pair: pair[0], matches)
        self.tok2id = {match.group(1):id for (match, id) in matches}
        pass


    def __repr__(self):
        return "<CNGraph>"

    @staticmethod
    def read(path: str):
        if os.path.isdir(path):
            return CNGraph.from_pykeen(path)
        with open(path, 'r') as r:
            return CNGraph.from_json(json.load(r))

    @staticmethod
    def from_pykeen(pykeen_dir: str):
        pykeen_dir  = pathlib.Path(pykeen_dir)
        # relation_df = pd.read_csv(pykeen_dir.join('relation_to_id.tsv.gz'), compression='gzip', delimiter='\t')

        entity_df   = pd.read_csv(pykeen_dir.joinpath('entity_to_id.tsv.gz'), compression='gzip', delimiter='\t')
        raw_uris = entity_df.label
        raw_ids = entity_df.id.apply(int)
        uri2id = dict(zip(raw_uris, raw_ids))
        id2uri = dict(zip(raw_ids, entity_df.label))

        del entity_df

        edge_df = pd.read_csv(pykeen_dir.joinpath('numeric_triples.tsv.gz'), compression='gzip', delimiter='\t')
        heads = edge_df['head'].apply(int) # head is also a DF method. Use [] to circumvent that
        tails = edge_df['tail'].apply(int)  # Same for tail
        rels = 2 * edge_df.relation.apply(int)
        inv_rels = rels + 1
        adj = defaultdict(list)
        for (head, tail, rel, inv_rel) in zip(heads, tails, rels, inv_rels):
            adj[head].append((tail, rel))
            adj[tail].append((head, inv_rel))
        return CNGraph(
            uri2id=uri2id,
            id2uri=id2uri,
            adj=adj
        )

    @staticmethod
    def from_json(json_data: Dict[str, Any]):
        id2uri = {int(k):v for k,v in json_data['id2uri'].items()}
        uri2id =  {v:k for k,v in id2uri.items()}
        adj = {int(k):v for k,v in json_data['adj'].items()}
        return CNGraph(uri2id=uri2id, id2uri=id2uri, adj=adj)
