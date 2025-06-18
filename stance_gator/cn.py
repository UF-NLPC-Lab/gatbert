# STL
import json
import pickle
import csv
from collections import defaultdict, Counter
from typing import Dict, Set, Tuple
import functools
# 3rd Party
from tqdm import tqdm
# Local
from .data import extract_cn_baseword

@functools.cache
def load_syn_map(assertions_or_json_path) -> Dict[str, Dict[str, str]]:
    with open(assertions_or_json_path, 'r') as r:
        try:
            return json.load(r)
        except json.JSONDecodeError:
            pass
    adj = defaultdict(lambda: defaultdict(Counter))

    def row_f(row):
        head, tail = row[2:4]
        head_comps = head.split('/')
        tail_comps = tail.split('/')
        head_lang = head_comps[2]
        tail_lang = tail_comps[2]
        head_baseword = extract_cn_baseword(head)
        tail_baseword = extract_cn_baseword(tail)
        return head_lang, head_baseword, tail_lang, tail_baseword

    with open(assertions_or_json_path) as r:
        reader = csv.reader(r, delimiter='\t')
        rows = filter(lambda row: row[1] == '/r/Synonym', tqdm(reader))
        rows = map(row_f, rows)
        rows = filter(lambda row: row[0] != row[2], rows)
        for (head_lang, head_baseword, tail_lang, tail_baseword) in rows:
            if head_lang == 'en':
                adj[tail_lang][head_baseword].update([tail_baseword])
            elif tail_lang == 'en':
                adj[head_lang][tail_baseword].update([head_baseword])
    adj = {k:{k2:c.most_common(1)[0][0] for k2,c in v.items()} for k,v in tqdm(adj.items())}
    return adj
 

class CN:

    @staticmethod
    def load(assertions_or_pickle_path):
        try:
            with open(assertions_or_pickle_path, 'rb') as r:
                return pickle.load(r)
        except pickle.UnpicklingError:
            return CN(assertions_or_pickle_path)

    def __init__(self, assertions_path):
        node2id = defaultdict(lambda: len(node2id))
        relation2id = defaultdict(lambda: len(relation2id))

        syn_map = defaultdict(lambda: defaultdict(set))
        adj = defaultdict(list)
        rev_adj = defaultdict(list)
        
        triples = []

        # Deterministic as long as you use the same assertions path
        with open(assertions_path, 'r') as r:
            reader = csv.reader(r, delimiter='\t')
            rows = list(reader)
        for row in tqdm(rows):
            relation_str, head, tail = row[1:4]

            head_lang = head.split('/')[2]
            tail_lang = tail.split('/')[2]
            head_str = extract_cn_baseword(head)
            tail_str = extract_cn_baseword(tail)

            if (head_lang == 'en' and tail_lang == 'en'):
                # Only english-english edges are used for the RGCN
                head_id = node2id[head_str]
                rel_id = relation2id[relation_str]
                tail_id = node2id[tail_str]
                adj[head_id].append((rel_id, tail_id))
                rev_adj[tail_id].append((rel_id, head_id))
                triples.append( [head_id, rel_id, tail_id] )
            # Cross-lingual edges are only used for synonym lookups
            elif head_lang == 'en':
                head_id = node2id[head_str]
                assert relation_str == '/r/Synonym'
                syn_map[tail_lang][tail_str].add(head_id)
            else:
                assert tail_lang == 'en'
                assert relation_str == '/r/Synonym'
                syn_map[head_lang][head_str].add(tail_id)

        self.node2id = dict(node2id)
        self.id2node = {v:k for k,v in self.node2id.items()}
        self.relation2id = dict(relation2id)

        self.syn_map: Dict[Tuple[str, str], Set[int]] = {(lang, lemma): syn_map[lang][lemma] for lang, lang_map in syn_map.items() for lemma in lang_map}

        self.adj = dict(adj)
        self.rev_adj = dict(rev_adj)

        self.triples = triples
        """
        List of triples
        """
