import csv
from collections import defaultdict

from .data import pretokenize_cn_uri

class CN:
    def __init__(self, assertions_path):
        node2id = defaultdict(lambda: len(node2id))
        relation2id = defaultdict(lambda: len(relation2id))
        
        edges = []

        # Deterministic as long as you use the same assertions path
        with open(assertions_path, 'r') as r:
            reader = csv.reader(r, delimiter='\t')
            for row in reader:
                relation_str = row[1]
                head_str = pretokenize_cn_uri(row[2])
                tail_str = pretokenize_cn_uri(row[3])
                head_id = node2id[head_str]
                rel_id = relation2id[relation_str]
                tail_id = node2id[tail_str]
                edges.append( [head_id, rel_id, tail_id] )

        self.node2id = dict(node2id)
        self.id2node = {v:k for k,v in self.node2id.items()}
        self.relation2id = dict(relation2id)
        self.edges = edges
