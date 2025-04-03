# STL
import argparse
import os
import pathlib
from collections import Counter
import gzip
import csv
# 3rd Party
import spacy
# Local
from .graph import GraphPaths, read_entitites, read_adj_mat, open_gzip_or_plain
from .utils import exists_gzip_or_plain
from .data import parse_vast

def main(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--vast", type=pathlib.Path, nargs="+", metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("-i", type=pathlib.Path, metavar="graph_in_dir/", required=True)
    parser.add_argument("-o", type=pathlib.Path, metavar="out_dir/", required=True)
    args = parser.parse_args(raw_args)

    # FIXME: Make configurable
    min_freq = 2
    max_seeds = 5000

    tags = {"PROPN", "NOUN", "ADJ", "ADV"}

    tagger = spacy.load("en_core_web_sm")

    tok_counts = Counter()
    for corpus in args.vast:
        for sample in parse_vast(corpus):
            spacy_doc = tagger(sample.context)
            tokens = [str(spacy_doc[i]) for i in range(len(spacy_doc) - 1)]
            tok_counts.update(tokens)
    tokens = [tok
              for tok,freq in tok_counts.most_common(max_seeds)
              if freq >= min_freq \
              and "_" not in tok \
              and all(spacy_tok.pos_ in tags for spacy_tok in tagger(tok))
    ]
    tok2id = {tok:i for i,tok in enumerate(tokens)}

    in_paths = GraphPaths(args.i)

    old2new = {}

    for (entity, old_id) in read_entitites(in_paths.entities_path).items():
        entity = entity.split('/')[3] if entity.startswith('/') else entity
        if entity in tok2id:
            old2new[old_id] = tok2id[entity]

    os.makedirs(args.o, exist_ok=True)
    out_paths = GraphPaths(args.o)
    # with gzip.open(out_paths.entities_path, "wb") as w:
    with open(out_paths.entities_path, "w") as w:
        writer = csv.DictWriter(w, fieldnames=["id", "label"], delimiter='\t')
        writer.writeheader()
        for (tok, id) in tok2id.items():
            writer.writerow({"id": id, "label": tok})

    def convert_triples(old_path, new_path):
        def map_row(row):
            row["head"]     = int(row["head"])
            row["tail"]     = int(row["tail"])
            if row["head"] not in old2new or row["tail"] not in old2new:
                return None
            row["head"] = old2new[row['head']]
            row["tail"] = old2new[row['tail']]
            row["relation"] = int(row["relation"])
            return row
        with open_gzip_or_plain(old_path) as r:
            reader = csv.DictReader(r, delimiter='\t')
            cols = reader.fieldnames
            rows = map(map_row, reader)
            rows = list(filter(lambda row: row, rows))
        # with gzip.open(new_path, "wb") as w:
        with open(new_path, "w") as w:
            writer = csv.DictWriter(w, fieldnames=cols, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    convert_triples(in_paths.triples_path, out_paths.triples_path)
    if exists_gzip_or_plain(in_paths.bert_triples_path):
        convert_triples(in_paths.bert_triples_path, out_paths.bert_triples_path)

    with open_gzip_or_plain(in_paths.relations_path) as r, gzip.open(out_paths.relations_path, 'wb') as w:
        w.write(r.read().encode())
    return

if __name__ == "__main__":
    main()