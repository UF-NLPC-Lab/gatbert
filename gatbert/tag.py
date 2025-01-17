"""
Tag a set of stance files with CN graph nodes
"""
# STL
import json
import argparse
import sys
import csv
from typing import List
# Local
from .constants import TOKEN_TO_KB_RELATION_ID, DEFAULT_MAX_DEGREE, NodeType
from .data import parse_ez_stance, PretokenizedSample, get_default_pretokenize
from .graph_sample import GraphSample
from .graph import CNGraph

def tag(sample: PretokenizedSample, graph: CNGraph, max_hops: int = 1, max_degree: int = DEFAULT_MAX_DEGREE) -> GraphSample:
    """
    Args:
        sample:
        max_hops:
        max_degree:
            If a node's degree exceeds this size, don't explore its neighborhood
    """

    builder = GraphSample.Builder(stance=sample.stance)

    frontier = set()
    def make_seed_dict(tokens: List[str]):
        rval = []
        for token in tokens:
            seeds = []
            kb_id = graph.tok2id.get(token.lower())
            if kb_id is not None:
                frontier.add(kb_id)
                seeds.append(kb_id)
            rval.append((token, seeds))
        return rval
    builder.add_seeds(
        make_seed_dict(sample.target),
        make_seed_dict(sample.context)
    )

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

def main(raw_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ezstance", type=str, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("--graph",    type=str, metavar="graph.json", help="File containing graph data written with .extract_cn")
    parser.add_argument("--max-degree", type=int, metavar=DEFAULT_MAX_DEGREE, default=DEFAULT_MAX_DEGREE, help="If a node's degree exceeds this size, don't explore its neighborhood")
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
    sample_gen = map(lambda s: tag(s, graph, max_degree=args.max_degree), sample_gen)
    sample_gen = map(lambda s: s.to_row(), sample_gen)
    with open(args.o, 'w', encoding='utf-8') as w:
        csv.writer(w, delimiter='\t').writerows(sample_gen)

if __name__ == "__main__":
    main(sys.argv[1:])