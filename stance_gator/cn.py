# STL
import csv
from collections import defaultdict
# 3rd Party
from tqdm import tqdm
# Local
from .data import extract_cn_baseword

class CN:
    def __init__(self, assertions_path):
        node2id = defaultdict(lambda: len(node2id))
        relation2id = defaultdict(lambda: len(relation2id))
        
        edges = []

        # Deterministic as long as you use the same assertions path
        with open(assertions_path, 'r') as r:
            reader = csv.reader(r, delimiter='\t')
            rows = list(reader)
        for row in tqdm(rows):
            relation_str = row[1]
            head_str = extract_cn_baseword(row[2])
            tail_str = extract_cn_baseword(row[3])
            head_id = node2id[head_str]
            rel_id = relation2id[relation_str]
            tail_id = node2id[tail_str]
            edges.append( [head_id, rel_id, tail_id] )

        self.node2id = dict(node2id)
        self.id2node = {v:k for k,v in self.node2id.items()}
        self.relation2id = dict(relation2id)
        self.edges = edges
