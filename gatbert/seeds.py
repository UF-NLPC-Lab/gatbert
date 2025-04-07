# STL
import sys
import argparse
import os
import shutil
import pathlib
from collections import Counter
import gzip
import csv
from typing import Iterable, List
# 3rd Party
import numpy as np
import torch
from tqdm import tqdm
import spacy
# Local
from .sample import Sample
from .graph import GraphPaths, read_entitites
from .utils import GzipWrapper, open_gzip_or_plain, Dictionary
from .data import parse_vast

def get_seeds(samples: Iterable[Sample],
              min_df = 2,
              max_df_frac = 0.5,
              max_seeds = 5000) -> List[str]:
    tags = {"PROPN", "NOUN", "ADJ", "ADV"}

    tagger = spacy.load("en_core_web_sm")

    gdict = Dictionary()
    # TODO: Include targets as well?
    samples = [sample.context for sample in samples]
    for sample in tqdm(samples):
        spacy_doc = tagger(sample)
        tokens = [str(spacy_doc[i]) for i in range(len(spacy_doc) - 1)]
        gdict.update(tokens)
    freq_tokens = gdict.filter_extremes(no_below=min_df, no_above=max_df_frac, keep_n=max_seeds)
    seeds = set(tok.lower() for tok in freq_tokens if "_" not in tok and all(spacy_tok.pos_ in tags for spacy_tok in tagger(tok)) )
    seeds = sorted(seeds)
    return seeds


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vast", type=pathlib.Path, nargs="+", metavar="input.csv", help="File containing stance data from the VAST dataset")
    parser.add_argument("-o", type=pathlib.Path, default="seeds.txt", metavar="seeds.txt", help="Output file")
    args = parser.parse_args(raw_args)

    # TODO: Support other corpora
    parse_fn = parse_vast

    def sample_gen(corpus_paths):
        for p in corpus_paths:
            yield from parse_fn(p)
    seeds = get_seeds(sample_gen(args.vast))
    with open(args.o, 'w') as w:
        print(w.write("\n".join(seeds)))
    print(f"Wrote {len(seeds)} seeds to {args.o}")

if __name__ == "__main__":
    main()