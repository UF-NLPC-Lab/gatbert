# STL
import argparse
import os
import shutil
import pathlib
from collections import Counter
import gzip
import csv
# 3rd Party
from tqdm import tqdm
import spacy
# Local
from .graph import GraphPaths, read_entitites
from .utils import GzipWrapper, open_gzip_or_plain
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
        samples = list(parse_vast(corpus))
        for sample in tqdm(samples):
            spacy_doc = tagger(sample.context)
            tokens = [str(spacy_doc[i]) for i in range(len(spacy_doc) - 1)]
            tok_counts.update(tokens)
        samples = None
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
    with gzip.open(out_paths.entities_path, "wb") as w:
        writer = csv.DictWriter(GzipWrapper(w), fieldnames=["id", "label"], delimiter='\t')
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
        with gzip.open(new_path, "wb") as w:
            writer = csv.DictWriter(GzipWrapper(w), fieldnames=cols, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    convert_triples(in_paths.triples_path, out_paths.triples_path)
    if os.path.exists(in_paths.bert_triples_path):
        convert_triples(in_paths.bert_triples_path, out_paths.bert_triples_path)

    shutil.copy(in_paths.relations_path, out_paths.relations_path)

if __name__ == "__main__":
    main()