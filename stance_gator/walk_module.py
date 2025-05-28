from __future__ import annotations
from collections import namedtuple, defaultdict
from itertools import product
import csv
# 3rd Party
import networkx as nx
import pathlib
import torch
import numpy as np
from tqdm import tqdm
# Local
from .base_module import StanceModule
from .cn import CN
from .encoder import Encoder, keyed_pad, keyed_scalar_stack
from .sample import Sample
from .data import SPACY_PIPES, extract_lemmas, extract_cn_baseword
from .utils import time_block

class AltCN:
    def __init__(self, assertions_path):
        # Deterministic as long as you use the same assertions path
        with open(assertions_path, 'r') as r:
            reader = csv.reader(r, delimiter='\t')
            rows = list(reader)
        nx_edges = defaultdict(set)
        node2id = defaultdict(lambda: len(node2id))
        relation2id = defaultdict(lambda: len(relation2id))
        for row in tqdm(rows):
            relation_str, head, tail = row[1:4]
            head_lang = head.split('/')[2]
            tail_lang = tail.split('/')[2]
            assert head_lang == 'en' and tail_lang == 'en'
            head_str = extract_cn_baseword(head)
            tail_str = extract_cn_baseword(tail)
            head_id = node2id[head_str]
            rel_id = relation2id[relation_str]
            tail_id = node2id[tail_str]
            nx_edges[head_id, tail_id].add(rel_id)

        self.node2id = dict(node2id)
        self.id2node = {v:k for k,v in self.node2id.items()}
        self.relation2id = dict(relation2id)

        self.G = nx.DiGraph()
        with time_block("label loop"):
            for (head, tail), rels in nx_edges.items():
                for_rels = list(rels)
                back_rels = [(rel_id + len(self.relation2id)) % (2 * len(self.relation2id)) for rel_id in for_rels]
                self.G.add_edge(head, tail, rel=for_rels)
                self.G.add_edge(tail, head, rel=back_rels)
        self.paths = dict()

    def get_shortest_path(self, head, tail):
        if (head, tail) in self.paths:
            return self.paths[head, tail]

        if head == tail:
            rel_id = self.relation2id['/r/Synonym']
            path = [(rel_id, tail)]
            rev_path = [(rel_id + len(self.relation2id), tail)]
            self.paths[head, tail] = path
            self.paths[tail, head] = rev_path
            return self.paths[head, tail]

        try:
            path = nx.shortest_path(self.G, source=head, target=tail)
        except nx.exception.NetworkXNoPath:
            self.paths[head, tail] = None
            self.paths[tail, head] = None
            return self.paths[head, tail]

        assert path[0] == head
        assert path[-1] == tail
        aug_path = []
        rev_aug_path = []
        prev = head
        for hop in path[1:]:
            # Just take the first relation if there are multiple
            rel = self.G.edges[prev, hop]['rel'][0]
            aug_path.append((rel, hop))

            rev_rel = (rel + len(self.relation2id)) % (2 * len(self.relation2id))
            rev_aug_path.insert(0, (rev_rel, prev))
            prev = hop

        for (i, (_, hop)) in enumerate(aug_path):
            self.paths[head, hop] = aug_path[:i+1]
        for (i, (_, hop)) in enumerate(rev_aug_path):
            self.paths[tail, hop] = rev_aug_path[:i+1]

        return self.paths[head, tail]

    def viz_path(self, path):
        to_print = []
        id2rel = {v:k for k,v in self.relation2id.items()}
        for rel, node in path:
            node_label = self.id2node[node]
            if rel >= len(id2rel):
                rel_label = id2rel[rel - len(id2rel)] + "/rev"
            else:
                rel_label = id2rel[rel]
            to_print.append((node_label, rel_label))
        return to_print

class WalkModule(StanceModule):
    def __init__(self,
                cn_path: pathlib.Path,
                feature_size: int = 64,
                max_length: int = 10,
                **parent_kwargs
                ):
        super().__init__(**parent_kwargs)

        self.cn = AltCN(cn_path)
        self.max_length = max_length

        self.relation_mat = torch.nn.Parameter(torch.empty(len(self.cn.relation2id) * 2, feature_size))
        torch.nn.init.xavier_normal_(self.relation_mat)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(feature_size, feature_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_size, len(self.stance_enum), bias=True)
        )
        self.ce_func = torch.nn.CrossEntropyLoss()
        self.__encoder = self.Encoder(self)

    @property
    def encoder(self):
        return self.__encoder

    Output = namedtuple("JointOutput", field_names=["logits", "loss"])

    def forward(self, input_ids, mask, labels=None):
        relation_embeds = input_ids.to(self.relation_mat.dtype) @ self.relation_mat
        mask_view = mask.view(*mask.shape, 1)
        relation_embeds = relation_embeds * mask_view

        path_sums = torch.sum(relation_embeds, dim=1)
        num_paths = torch.sum(mask, dim=-1, keepdim=True)
        mean = path_sums / (num_paths + 1e-6)
        logits = self.classifier(mean)
        loss = None
        if labels is not None:
            loss = self.ce_func(logits, labels)
        return self.Output(logits, loss)

    class Encoder(Encoder):
        def __init__(self, wm: WalkModule):
            self.wn = wm
            self.cn = wm.cn
            self.feature_size = 2 * len(self.cn.relation2id)
            self.empty_count = 0

        def get_nodes(self, pipeline, text: str):
            lemmas = extract_lemmas(pipeline, text)
            return set([self.cn.node2id[l] for l in lemmas if l in self.cn.node2id])

        def encode(self, sample: Sample):
            if sample.is_split_into_words:
                context = " ".join(sample.context)
                target = " ".join(sample.target)
            else:
                context = sample.context
                target = sample.target
            lang = sample.lang or 'en'
            assert lang == 'en'
            pipeline = SPACY_PIPES[lang]

            target_nodes = self.get_nodes(pipeline, target)
            context_nodes = self.get_nodes(pipeline, context)
            paths = []
            for (target_lemma, context_lemma) in product(target_nodes, context_nodes):
                rel_counts = np.zeros(self.feature_size, dtype=np.long)
                path = self.cn.get_shortest_path(target_lemma, context_lemma)
                if path:
                    for (rel, _) in path:
                        rel_counts[rel] += 1
                    paths.append(rel_counts)
            if not paths:
                self.empty_count += 1
                feature_vec = torch.empty([0, self.feature_size])
                return None
            else:
                feature_vec = torch.tensor(np.stack(paths))
            return {
                "input_ids": feature_vec,
                'labels': torch.tensor([sample.stance.value])
            }
        
        def collate(self, samples):
            padded = keyed_pad(samples, "input_ids", -1)
            mask = torch.logical_not(torch.all(padded == -1, dim=-1))
            return {"input_ids": padded, "mask": mask, "labels": keyed_scalar_stack(samples, 'labels')}

