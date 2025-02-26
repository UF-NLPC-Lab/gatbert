"""
Tag a set of stance files with CN graph nodes
"""
# STL
import json
import argparse
import sys
import csv
from typing import List
from collections import defaultdict, OrderedDict
from itertools import islice
# Local
from .data import parse_ez_stance, PretokenizedSample, get_default_pretokenize
from .graph_sample import GraphSample
from .graph import CNGraph

def tag(sample: PretokenizedSample, graph: CNGraph) -> GraphSample:
    """
    Args:
        sample:
        max_hops:
        max_degree:
            If a node's degree exceeds this size, don't explore its neighborhood
    """

    frontier = set()
    def make_seed_dict(tokens: List[str]):
        rval = OrderedDict()
        for token in tokens:
            seeds = []
            kb_id = graph.tok2id.get(token.lower())
            if kb_id is not None:
                frontier.add(kb_id)
                seeds.append(kb_id)
            rval[token] = seeds
        return rval

    target_seed_dict = make_seed_dict(sample.target)
    context_seed_dict = make_seed_dict(sample.context)

    flatten = lambda sd: set(s for l in sd.values() for s in l)

    target_seeds = flatten(target_seed_dict)
    context_seeds = flatten(context_seed_dict)

    def get_first_hops(seeds):
        hop_dict = defaultdict(set)
        for seed in seeds:
            for (neighbor, _) in graph.adj.get(seed, []):
                hop_dict[neighbor].add(seed)
        return hop_dict

    target_1_hop = get_first_hops(target_seeds)
    target_2_hops = get_first_hops(target_1_hop)

    context_1_hop = get_first_hops(context_seeds)
    context_2_hops = get_first_hops(context_1_hop)

    def add_to(s, el):
        s.add(el)

    kept_forward = set()
    for (head, predecessors) in filter(lambda p: p[0] in context_seeds, target_2_hops.items()):
        add_to(kept_forward, head)
        for p in predecessors:
            add_to(kept_forward, p)
    context_or_kept = kept_forward | context_seeds
    for (head, predecessors) in filter(lambda p: p[0] in context_or_kept, target_1_hop.items()):
        add_to(kept_forward, head)
        for p in predecessors:
            add_to(kept_forward, p)

    kept_backward = set()
    for (head, predecessors) in filter(lambda p: p[0] in target_seeds, context_2_hops.items()):
        add_to(kept_backward, head)
        for p in predecessors:
            add_to(kept_backward, p)
    target_or_kept = kept_backward | target_seeds
    for (head, predecessors) in filter(lambda p: p[0] in target_or_kept, context_1_hop.items()):
        add_to(kept_backward, head)
        for p in predecessors:
            add_to(kept_backward, p)

    kept = kept_forward | kept_backward
    def prune_seeds(seed_dict):
        toks = list(seed_dict)
        for tok in toks:
            seed_dict[tok] = [s for s in seed_dict[tok] if s in kept]
    prune_seeds(target_seed_dict)
    prune_seeds(context_seed_dict)

    builder = GraphSample.Builder(stance=sample.stance)
    builder.add_seeds(target_seed_dict, context_seed_dict)

    flatten_lists = lambda seed_dict: [v for vlist in seed_dict.values() for v in vlist]
    frontier = flatten_lists(target_seed_dict) + flatten_lists(context_seed_dict)
    visited = set()
    while frontier:
        head = frontier.pop(0)
        if head in visited:
            continue
        visited.add(head)
        for (tail, relation) in filter(lambda pair: pair[0] in kept, graph.adj.get(head, [])):
            builder.add_kb_edge(head, tail, relation)
            frontier.append(tail)

    graph_sample = builder.build()
    # Replace all KB IDs with KB uris now
    for i in range(len(graph_sample.kb)):
        graph_sample.kb[i] = graph.id2uri[graph_sample.kb[i]]
    return graph_sample

def main(raw_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ezstance", type=str, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("--graph",    type=str, metavar="graph.json", help="File containing graph data written with .extract_cn")
    # parser.add_argument("--max-degree", type=int, metavar=DEFAULT_MAX_DEGREE, default=DEFAULT_MAX_DEGREE, help="If a node's degree exceeds this size, don't explore its neighborhood")
    parser.add_argument("-o",         type=str, required=True, metavar="output_file.tsv", help="TSV file containing samples with associated CN nodes")
    args = parser.parse_args(raw_args)

    if sum([bool(args.ezstance), bool(args.vast), bool(args.semeval)]) != 1:
        print("Must select exactly one of --ezstance, --vast, or --semeval", file=sys.stderr)
        sys.exit(1)
    if args.ezstance:
        sample_gen = parse_ez_stance(args.ezstance)
    elif args.vast:
        raise RuntimeError("--vast not yet supported")
    else:
        raise RuntimeError("--semeval not yet supported")

    with open(args.graph, 'r', encoding='utf-8') as r:
        graph = CNGraph.from_json(json.load(r))

    sample_gen = map(get_default_pretokenize(), sample_gen)
    sample_gen = map(lambda s: tag(s, graph), sample_gen)
    sample_gen = map(lambda s: s.to_row(), sample_gen)
    sample_gen = islice(sample_gen, 4)
    with open(args.o, 'w', encoding='utf-8') as w:
        csv.writer(w, delimiter='\t').writerows(sample_gen)

if __name__ == "__main__":
    main(sys.argv[1:])