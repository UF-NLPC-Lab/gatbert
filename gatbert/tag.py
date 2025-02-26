"""
Tag a set of stance files with CN graph nodes
"""
# STL
import json
import argparse
import sys
import csv
from typing import List, Callable
from collections import defaultdict
from itertools import islice
# Local
from .constants import DEFAULT_MAX_DEGREE
from .data import parse_ez_stance, PretokenizedSample, get_default_pretokenize
from .graph_sample import GraphSample
from .graph import CNGraph

def make_seed_dict(graph: CNGraph, tokens: List[str]):
    rval = []
    for token in tokens:
        seeds = []
        kb_id = graph.tok2id.get(token.lower())
        if kb_id is not None:
            seeds.append(kb_id)
        rval.append((token, seeds))
    return rval

def naive_sample(sample: PretokenizedSample, graph: CNGraph, max_hops: int = 1, max_degree: int = DEFAULT_MAX_DEGREE) -> GraphSample:
    """
    Just take the seed concepts that match tokens in the sample,
    and nodes that neighbor those seeds.

    If a node's out-degree is >= max_degree, we don't explore its neighborhood
    (still keep the node itself though).
    """

    builder = GraphSample.Builder(stance=sample.stance)

    target_seeds = make_seed_dict(graph, sample.target)
    context_seeds = make_seed_dict(graph, sample.context)
    builder.add_seeds(
        target_seeds,
        context_seeds
    )

    frontier = {node for (_, node_list) in target_seeds + context_seeds for node in node_list}
    visited = set()
    for _ in range(max_hops):
        visited |= frontier
        last_frontier = frontier
        frontier = set()
        for head_kb_id in last_frontier:
            outgoing = graph.adj.get(head_kb_id, [])
            if len(outgoing) <= max_degree:
                for (tail_kb_id, relation_id) in outgoing:
                    builder.add_kb_edge(head_kb_id, tail_kb_id, relation_id)
                    frontier.add(tail_kb_id)
        frontier -= visited

    for head_kb_id in frontier:
        for (tail_kb_id, relation_id) in graph.adj[head_kb_id]:
            if tail_kb_id in frontier:
                builder.add_kb_edge(head_kb_id, tail_kb_id, relation_id)

    graph_sample = builder.build()
    # Replace all KB IDs with KB uris now
    for i in range(len(graph_sample.kb)):
        graph_sample.kb[i] = graph.id2uri[graph_sample.kb[i]]
    return graph_sample

def bridge_sample(sample: PretokenizedSample, graph: CNGraph) -> GraphSample:
    """
    Sample for CN nodes that act as "bridges" between target tokens and context tokens.

    See https://aclanthology.org/2021.findings-acl.278/ for reference.
    We extract seed concepts for both the target and context, and explore one hop beyond those.
    We only keep nodes that are part of a 0-, 1-, or 2-hop path between target seeds and context seeds.
    """

    target_seed_dict = make_seed_dict(graph, sample.target)
    context_seed_dict = make_seed_dict(graph, sample.context)

    flatten = lambda sd: set(node for (_, node_list) in sd for node in node_list)
    target_seeds = flatten(target_seed_dict)
    context_seeds = flatten(context_seed_dict)
    common_seeds = target_seeds & context_seeds
    target_seeds -= common_seeds
    context_seeds -= common_seeds

    def get_hops(seeds):
        hop_dict = defaultdict(set)
        for seed in seeds:
            for (neighbor, _) in graph.adj.get(seed, []):
                hop_dict[neighbor].add(seed)
        return hop_dict

    def trace_hops(required, source):
        to_keep = set()
        for (tail, predecessors) in filter(lambda p: p[0] in required, source.items()):
            to_keep.add(tail)
            to_keep.update(predecessors)
        return to_keep

    target_or_context = target_seeds | context_seeds
    common_1_hop = get_hops(common_seeds)
    common_2_hop = get_hops(common_1_hop)
    kept_middle = trace_hops(target_or_context, common_2_hop) | trace_hops(target_or_context, common_1_hop)

    context_or_common = context_seeds | common_seeds
    target_1_hop = get_hops(target_seeds)
    target_2_hops = get_hops(target_1_hop)
    kept_forward = trace_hops(context_or_common, target_2_hops) | trace_hops(context_or_common, target_1_hop)

    target_or_common = target_seeds | common_seeds
    context_1_hop = get_hops(context_seeds)
    context_2_hops = get_hops(context_1_hop)
    kept_backward = trace_hops(target_or_common, context_2_hops) | trace_hops(target_or_common, context_1_hop)

    kept = kept_forward | kept_backward | kept_middle | common_seeds
    def prune_seeds(seed_dict):
        for i in range(len(seed_dict)):
            seed_dict[i] = (seed_dict[i][0], [s for s in seed_dict[i][1] if s in kept])
    prune_seeds(target_seed_dict)
    prune_seeds(context_seed_dict)

    builder = GraphSample.Builder(stance=sample.stance)
    builder.add_seeds(target_seed_dict, context_seed_dict)

    frontier = list(kept)
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

    def get_tag_func(name) -> Callable[[PretokenizedSample, CNGraph], GraphSample]:
        if name == "bridge":
            return bridge_sample
        elif name == "naive":
            return naive_sample
        else:
            raise ValueError(f"Invalid --sample {name}")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ezstance", type=str, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("--graph",    type=str, metavar="graph.json", help="File containing graph data written with .extract_cn")

    parser.add_argument("--sample", type=get_tag_func, default=bridge_sample, metavar="bridge|naive", help="How to sample nodes from the CN graph")
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
    tag_func = args.sample

    sample_gen = map(get_default_pretokenize(), sample_gen)
    sample_gen = map(lambda s: tag_func(s, graph), sample_gen)
    sample_gen = map(lambda s: s.to_row(), sample_gen)
    with open(args.o, 'w', encoding='utf-8') as w:
        csv.writer(w, delimiter='\t').writerows(sample_gen)

if __name__ == "__main__":
    main(sys.argv[1:])