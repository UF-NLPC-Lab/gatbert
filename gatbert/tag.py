"""
Tag a set of stance files with CN graph nodes
"""
# STL
import json
import argparse
import sys
import csv
from itertools import islice
# Local
from .constants import TOKEN_TO_KB_RELATION_ID, DEFAULT_MAX_DEGREE
from .data import parse_ez_stance, PretokenizedSample, get_default_pretokenize, GraphSample
from .graph import CNGraph

def tag(sample: PretokenizedSample, graph: CNGraph, max_hops: int = 1, max_degree: int = DEFAULT_MAX_DEGREE) -> GraphSample:
    """
    Args:
        sample:
        max_hops:
        max_degree:
            If a node's degree exceeds this size, don't explore its neighborhood
    """
    token_nodes = [*sample.target, *sample.context]
    nodes = [*token_nodes]
    kb2arrind = dict()
    def get_node_index(kb_id: int):
        if kb_id not in kb2arrind:
            kb2arrind[kb_id] = len(nodes)
            nodes.append(kb_id)
        return kb2arrind[kb_id]

    # TODO: Can we use a list for this? Do we need to worry about duplicates?
    # (head_ind, tail_ind, internal_relation_id)
    edges = set()

    frontier = set()
    for (token_ind, token) in enumerate(map(lambda t: t.lower(), token_nodes)):
        kb_id = graph.tok2id.get(token)
        if kb_id is None:
            continue
        kb_ind = get_node_index(kb_id)
        edges.add( (token_ind, kb_ind, TOKEN_TO_KB_RELATION_ID) )
        edges.add( (kb_ind, token_ind, TOKEN_TO_KB_RELATION_ID) )
        frontier.add(kb_id)

    visited = set()
    for _ in range(max_hops):
        visited |= frontier
        last_frontier = frontier
        frontier = set()
        for head_kb_id in last_frontier:
            head_arr_ind = kb2arrind[head_kb_id] # Safe to use direct dictionary lookup here instead of the function
            outgoing = graph.adj.get(head_kb_id, [])
            if len(outgoing) <= max_degree:
                for (tail_kb_id, relation_id) in outgoing:
                    tail_arr_ind = get_node_index(tail_kb_id)
                    edges.add((head_arr_ind, tail_arr_ind, relation_id))
                    frontier.add(head_kb_id)
                    frontier.add(tail_kb_id)
        frontier -= visited

    missing_edges = set()
    for head_kb_id in frontier:
        head_arr_ind = kb2arrind[head_kb_id]
        missing_edges.update((head_arr_ind, kb2arrind[tail_kb_id], relation_id) for (tail_kb_id, relation_id) in graph.adj[head_kb_id] if tail_kb_id in frontier)
    edges |= missing_edges

    n_context = len(sample.context)
    n_target = len(sample.target)
    # Replace all KB IDs with KB uris now
    for i in range(n_target + n_context, len(nodes)):
        nodes[i] = graph.id2uri[nodes[i]]

    return GraphSample(
        nodes=nodes,
        n_target=n_target,
        n_context=n_context,
        edges=list(edges),
        stance=sample.stance
    )

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