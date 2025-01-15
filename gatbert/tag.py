"""
Tag a set of stance files with CN graph nodes
"""
# STL
import argparse
import sys
from collections import defaultdict
# Local
from .constants import TOKEN_TO_KB_RELATION_ID
from .data import parse_ez_stance, Sample, PretokenizedSample, get_default_pretokenize
from .graph import CNGraph

def tag(sample: PretokenizedSample, graph: CNGraph, max_hops: int = 2):

    n_context = len(sample.context)
    n_target = len(sample.target)

    n_tokens = n_context + n_target

    token_nodes = [*sample.target, *sample.context]
    nodes = [*token_nodes]
    kb2arrind = dict()
    def get_node_index(kb_id: int):
        if kb_id not in kb2arrind:
            kb2arrind[kb_id] = len(nodes)
            nodes.append(kb_id)
        return kb2arrind[kb_id]

    # (head_ind, tail_ind, internal_relation_id)
    edges = set()

    frontier = set()
    for (token_ind, token) in enumerate(map(lambda t: t.lower(), token_nodes)):
        kb_id = graph.tok2id.get(token)
        if kb_id is None:
            continue
        kb_ind = get_node_index(kb_id)
        edges.append( (token_ind, kb_ind, TOKEN_TO_KB_RELATION_ID) )
        edges.append( (kb_ind, token_ind, TOKEN_TO_KB_RELATION_ID) )
        frontier.add(kb_id)

    visited = set()
    for _ in range(max_hops):
        visited |= frontier
        last_frontier = frontier
        frontier = set()
        for head_kb_id in last_frontier:
            head_arr_ind = kb2arrind[head_kb_id] # Safe to use direct dictionary lookup here instead of the function
            for (tail_kb_id, relation_id) in graph.adj[head_kb_id]:
                tail_arr_ind = get_node_index(tail_kb_id)
                edges.append(head_arr_ind, tail_arr_ind, relation_id)
                frontier.add(head_kb_id)
                frontier.add(tail_kb_id)
        frontier -= visited

    missing_edges = set()
    for head_kb_id in frontier:
        head_arr_ind = kb2arrind[head_kb_id]
        missing_edges.extend((head_arr_ind, kb2arrind[tail_kb_id], relation_id) for (tail_kb_id, relation_id) in graph.adj[head_kb_id] if tail_kb_id in frontier)
    edges |= missing_edges




def main(raw_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ezstance", type=str, metavar="input.csv", help="File containing stance data from the EZStance dataset")
    parser.add_argument("--vast",     type=str, metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("--semeval",  type=str, metavar="input.txt", help="File containing stance data from SemEval2016-Task6")
    parser.add_argument("--graph",    type=str, metavar="graph.json", help="File containing graph data written with .extract_cn")
    parser.add_argument("-o",         type=str, required=True, metavar="output_file.tsv", help="TSV file containing samples with associated CN nodes")
    args = parser.parse_args()

    if not any([args.ezstance, args.vast, args.semeval]):
        print("Must select one of --ezstance, --vast, or --semeval", file=sys.stderr)
        sys.exit(1)
    if args.ezstance:
        sample_gen = parse_ez_stance(args.ezstance)
    elif args.vast:
        raise RuntimeError("--vast not yet supported")
    else:
        raise RuntimeError("--semeval not yet supported")

    with open(args.graph, 'r', encoding='utf-8') as r:
        graph = CNGraph.from_json(r.read())

    sample_gen = map(get_default_pretokenize, sample_gen)
    sample_gen = map(lambda s: tag(s, graph), sample_gen)
    for s in sample_gen:
        pass

if __name__ == "__main__":
    main(sys.argv[1:])