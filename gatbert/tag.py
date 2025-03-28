"""
Tag a set of stance files with CN graph nodes
"""
# STL
from typing import Dict
import argparse
import sys
import csv
from typing import List
from collections import defaultdict
from tqdm import tqdm
# Local
from .constants import DEFAULT_MAX_DEGREE, CN_URI_PATT
from .data import parse_ez_stance, PretokenizedSample, get_default_pretokenize, parse_vast
from .graph_sample import GraphSample
from .graph import AdjMat, get_triples_path, get_bert_triples_path, read_adj_mat, update_adj_mat, read_entitites, get_entities_path, read_bert_adj_mat

def make_seed_dict(tok2id, tokens: List[str]):
    rval = []
    for token in tokens:
        seeds = []
        kb_id = tok2id.get(token.lower())
        if kb_id is not None:
            seeds.append(kb_id)
        rval.append((token, seeds))
    return rval

def naive_sample(sample: PretokenizedSample,
                  tok2id: Dict[str, int],
                  id2ent: Dict[int, str],
                  adj: AdjMat,
                 max_hops: int = 1,
                 max_degree: int = DEFAULT_MAX_DEGREE) -> GraphSample:
    """
    Just take the seed concepts that match tokens in the sample,
    and nodes that neighbor those seeds.

    If a node's out-degree is >= max_degree, we don't explore its neighborhood
    (still keep the node itself though).
    """

    builder = GraphSample.Builder(stance=sample.stance)

    target_seeds = make_seed_dict(tok2id, sample.target)
    context_seeds = make_seed_dict(tok2id, sample.context)
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
            outgoing = adj.get(head_kb_id, [])
            if len(outgoing) <= max_degree:
                for (tail_kb_id, relation_id) in outgoing:
                    builder.add_kb_edge(head_kb_id, tail_kb_id, relation_id)
                    frontier.add(tail_kb_id)
        frontier -= visited

    for head_kb_id in frontier:
        for (tail_kb_id, relation_id) in adj[head_kb_id]:
            if tail_kb_id in frontier:
                builder.add_kb_edge(head_kb_id, tail_kb_id, relation_id)

    graph_sample = builder.build()
    # Replace all KB IDs with KB uris now
    for i in range(len(graph_sample.kb)):
        graph_sample.kb[i] = id2ent[graph_sample.kb[i]]
    return graph_sample

def bridge_sample(sample: PretokenizedSample,
                  tok2id: Dict[str, int],
                  id2ent: Dict[int, str],
                  adj: AdjMat) -> GraphSample:
    """
    Sample for CN nodes that act as "bridges" between target tokens and context tokens.

    See https://aclanthology.org/2021.findings-acl.278/ for reference.
    We extract seed concepts for both the target and context, and explore one hop beyond those.
    We only keep nodes that are part of a 0-, 1-, or 2-hop path between target seeds and context seeds.
    """

    target_seed_dict = make_seed_dict(tok2id, sample.target)
    context_seed_dict = make_seed_dict(tok2id, sample.context)

    flatten = lambda sd: set(node for (_, node_list) in sd for node in node_list)
    target_seeds = flatten(target_seed_dict)
    context_seeds = flatten(context_seed_dict)
    common_seeds = target_seeds & context_seeds
    target_seeds -= common_seeds
    context_seeds -= common_seeds

    kept = set()
    def get_two_hops(seeds, query_set):
        hop_dict = defaultdict(set)
        frontier = []
        for seed in seeds:
            assert seed not in query_set
            for (neighbor, _) in adj.get(seed, []):
                hop_dict[neighbor].add(seed)
                frontier.append(neighbor)
                if neighbor in query_set:
                    kept.add(neighbor)
                    kept.add(seed)
        for node in frontier:
            for (neighbor, _) in adj.get(node, []):
                if neighbor in query_set:
                    kept.add(neighbor)
                    kept.add(node)
                    kept.update(hop_dict[node])
    get_two_hops(target_seeds, common_seeds | context_seeds)
    get_two_hops(context_seeds, common_seeds | target_seeds)
    get_two_hops(common_seeds,  target_seeds | context_seeds)
    kept.update(common_seeds)

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
        for (tail, relation) in filter(lambda pair: pair[0] in kept, adj.get(head, [])):
            builder.add_kb_edge(head, tail, relation)
            frontier.append(tail)

    graph_sample = builder.build()
    # Replace all KB IDs with KB uris now
    for i in range(len(graph_sample.kb)):
        graph_sample.kb[i] = id2ent[graph_sample.kb[i]]
    return graph_sample

def main(raw_args=None):

    def get_tag_func(name):
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
    parser.add_argument("--graph",    type=str, metavar="graph_dir/", required=True, help="Directory containing CN and bert triples")

    parser.add_argument("--bert-sim", action="store_true", help="Add BERT similarity edges")
    parser.add_argument("--bert-sim-thresh", type=float, default=0.5, help="Similarity threshold to include edge when using --bert-sim")
    parser.add_argument("--sample", type=get_tag_func, default=bridge_sample, metavar="bridge|naive", help="How to sample nodes from the CN graph")
    parser.add_argument("-o",         type=str, required=True, metavar="output_file.tsv", help="TSV file containing samples with associated CN nodes")
    args = parser.parse_args(raw_args)

    if sum([bool(args.ezstance), bool(args.vast), bool(args.semeval)]) != 1:
        print("Must select exactly one of --ezstance, --vast, or --semeval", file=sys.stderr)
        sys.exit(1)
    if args.ezstance:
        sample_gen = parse_ez_stance(args.ezstance)
    elif args.vast:
        sample_gen = parse_vast(args.vast)
    else:
        raise RuntimeError("--semeval not yet supported")

    def entity_to_tok(entity_id: str):
        if not entity_id.startswith('/'):
            toks = entity_id.split()
            return toks[0] if len(toks) == 1 else None
        match_obj = CN_URI_PATT.fullmatch(entity_id)
        return match_obj.group(1) if match_obj else None

    # Read graph data from disk
    ent2id = read_entitites(get_entities_path(args.graph))
    matches = map(lambda pair: (entity_to_tok(pair[0]), pair[1]), ent2id.items())
    tok2id= {tok:id for tok,id in matches if tok}

    id2ent = {v:k for k,v in ent2id.items()}
    adj = read_adj_mat(get_triples_path(args.graph))
    if args.bert_sim:
        bert_adj = read_bert_adj_mat(get_bert_triples_path(args.graph), sim_threshold=args.bert_sim_thresh)
        update_adj_mat(adj, bert_adj)

    tag_func = args.sample

    samples = list(map(get_default_pretokenize(), sample_gen))
    processed = []
    for sample in tqdm(samples):
        processed.append(tag_func(sample, tok2id, id2ent, adj).to_row())
    with open(args.o, 'w', encoding='utf-8') as w:
        csv.writer(w, delimiter='\t').writerows(processed)

if __name__ == "__main__":
    main(sys.argv[1:])