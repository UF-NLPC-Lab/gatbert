# STL
import os
import json
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import dataclasses
import pathlib
# 3rd Party
import pandas as pd
# Local

EdgeList = List[Tuple[int, int, int]]
NodeList = List[int]

@dataclasses.dataclass
class CNGraph:
    tok2id: Dict[str, int]
    id2uri: Dict[int, str]
    adj: Dict[int, List[Tuple[int, int]]]

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
        raw_toks = entity_df.label.apply(lambda l: l.split('/')[3])
        raw_ids = entity_df.id.apply(int)
        tok2id = dict(zip(raw_toks, raw_ids))
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
            tok2id=tok2id,
            id2uri=id2uri,
            adj=adj
        )

    @staticmethod
    def from_json(json_data: Dict[str, Any]):
        return CNGraph(
            tok2id=json_data['tok2id'],
            id2uri={int(k):v for k,v in json_data['id2uri'].items()},
            adj   ={int(k):v for k,v in json_data['adj'].items()}
        )
