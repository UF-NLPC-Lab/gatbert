# STL
import argparse
import pathlib
from collections import defaultdict
import os
import csv
# 3rd Party
from tqdm import tqdm
import spacy
# Local
from .graph import GraphPaths
from .encoder import pretokenize_cn_uri
from .data import parse_vast

def window_iter(seq, n=1):
    for i in range(0, len(seq) - n + 1):
        yield seq[i:i+n]

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-cn", type=pathlib.Path, metavar='assertions.tsv')
    parser.add_argument("--vast", type=pathlib.Path, nargs="+", metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("-o", type=pathlib.Path, required=True, metavar="out_graph_dir/", help="Output dir")


    args = parser.parse_args(raw_args)


    concepts = set()
    edges = []
    adj = defaultdict(set)

    cn_path = args.cn
    out_paths = GraphPaths(args.o)
    assert cn_path != out_paths.assertions_path
    os.makedirs(args.o, exist_ok=True)

    with open(args.cn, 'r') as r:
        reader = csv.reader(r, delimiter='\t')
        for row in reader:
            rel = row[1]

            # TODO: This is lossy as we're losing CN uri distinctions. Good or bad...?
            head = "_".join(pretokenize_cn_uri(row[2]))
            tail = "_".join(pretokenize_cn_uri(row[3]))

            edges.append((row[0], rel, head, tail, *row[4:]))

            adj[head].add(tail)
            adj[tail].add(head)
            concepts.add(head)
            concepts.add(tail)


    contexts = set()
    context_adj = set()
    targets = set()
    target_adj = set()

    tagger = spacy.load("en_core_web_sm")

    # TODO: Support other corpora
    parse_fn = parse_vast

    def find_seeds(text: str, seed_set, adj_set: set):
        tagged = tagger(text)
        toks = [tok.lemma_.lower() for tok in tagged]
        for window_size in range(1, 4):
            for window in window_iter(toks, window_size):
                ngram = "_".join(window)
                if ngram in concepts:
                    seed_set.add(ngram)
                    adj_set.add(ngram)
                    adj_set.update(adj[ngram])

    samples = list(s for p in args.vast for s in parse_fn(p))
    for sample in tqdm(samples):
        find_seeds(sample.context, contexts, context_adj)
        find_seeds(sample.target, targets, target_adj)

    def valid_edge(e):
        (head, tail) = e[2:4]
        return (head in contexts and tail in target_adj) or \
            (head in target_adj and head in contexts) or \
            (head in targets and tail in context_adj) or \
            (head in context_adj and tail in targets) or \
            (head in contexts and tail in contexts) or \
            (head in targets and tail in targets)
    with open(out_paths.assertions_path, 'w') as w:
        for (field1, rel, head, tail, *rem) in filter(valid_edge, edges):
            print(field1, rel, f"/c/en/{head}", f"/c/en/{tail}", *rem, sep='\t', file=w)

    target_and_context = contexts & targets
    target_or_context = sorted(contexts | targets)
    with open(out_paths.seeds_path, 'w') as w:
        for seed in target_or_context:
            if seed in target_and_context:
                print(seed, 1, 1, sep='\t', file=w)
            elif seed in targets:
                print(seed, 1, 0, sep='\t', file=w)
            else:
                print(seed, 0, 1, sep='\t', file=w)

if __name__ == "__main__":
    main()