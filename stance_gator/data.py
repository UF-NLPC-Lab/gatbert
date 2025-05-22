# STL
from __future__ import annotations
import json
import argparse
import csv
import os
from typing import Generator, Callable, Dict, Literal
import pathlib
from collections import defaultdict
# 3rd Party
import torch
import spacy
# Local
from .constants import TriStance, BiStance
from .sample import Sample


class MapDataset(torch.utils.data.Dataset):
    """
    In-memory dataset for stance samples
    """

    @staticmethod
    def from_dataset(ds: torch.utils.data.Dataset):
        return MapDataset(list(ds))

    def filter(self, pred) -> MapDataset:
        return MapDataset(list(filter(pred, self.__samples)))
    
    def map(self, f) -> MapDataset:
        return MapDataset(list(map(f, self.__samples)))

    def __init__(self, samples):
        self.__samples = list(samples)
    def __getitem__(self, key):
        return self.__samples[key]
    def __len__(self):
        return len(self.__samples)

def parse_ez_stance(csv_path) -> Generator[Sample, None, None]:
    strstance2 = {"FAVOR": TriStance.favor, "AGAINST": TriStance.against, "NONE": TriStance.neutral}
    def f(row):
        return Sample(row['Text'],
                      row['Target 1'],
                      strstance2[row['Stance 1']],
                      is_split_into_words=False)
    with open(csv_path, 'r', encoding='latin-1') as r:
        yield from map(f, csv.DictReader(r))

def parse_vast(csv_path) -> Generator[Sample, None, None]:
    strstance2enum = {
       "0": TriStance.against,
       "1": TriStance.favor,
       "2": TriStance.neutral
    }
    # TODO: Do we still want to use "post", or one of their preprocessed versions?
    with open(csv_path, 'r') as r:
        yield from map(lambda row: Sample(row['post'], row['topic_str'], strstance2enum[row['label']], is_split_into_words=False), csv.DictReader(r))

def parse_semeval(annotations_path) -> Generator[Sample, None, None]:
    raise NotImplementedError

def parse_xstance(jsonl_path) -> Generator[Sample, None, None]:
    stance_map = {
        "AGAINST": BiStance.against,
        "FAVOR": BiStance.favor
    }
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as r:
        for l in r:
            json_obj = json.loads(l)
            yield Sample(
                context=json_obj['comment'],
                target=json_obj['question'],
                stance=stance_map[json_obj['label']],
                is_split_into_words=False,
                lang=json_obj['language']
            )
    return samples

CorpusType = Literal['ezstance', 'semeval', 'vast', 'xstance']

StanceParser = Callable[[os.PathLike], Generator[Sample, None, None]]
"""
Function taking a file path and returning a generator of samples
"""

CORPUS_PARSERS: Dict[CorpusType, StanceParser] = {
    "ezstance": parse_ez_stance,
    "vast": parse_vast,
    "semeval": parse_semeval,
    "xstance": parse_xstance
}

def add_corpus_args(parser: argparse.ArgumentParser):
    for name in CORPUS_PARSERS:
        parser.add_argument(f"--{name}", type=pathlib.Path, metavar="data.(csv|jsonl)")

def get_sample_iter(args) -> Generator[Sample, None, None]:
    found_name = None
    found_iter = None
    for name, parse_fn in CORPUS_PARSERS.items():
        file_path = getattr(args, name)
        if file_path:
            if found_name:
                raise ValueError(f"Given both --{found_name} and --{name}")
            found_name = name
            found_iter = parse_fn(file_path)
    if found_iter:
        yield from found_iter
        return
    raise ValueError("Must provide one of " + ",".join([f"--{name}" for name in CORPUS_PARSERS]))

def load_parser(lang):
    raise ValueError(f"Unsupported language {lang}")

class __LazySpacyDict(dict):
    def __missing__(self, lang):
        if lang == 'en':
            self[lang] = spacy.load('en_core_web_sm')
        elif lang == 'de':
            self[lang] = spacy.load('de_core_news_sm')
        elif lang == 'it':
            self[lang] = spacy.load('it_core_news_sm')
        elif lang == 'fr':
            self[lang] = spacy.load('fr_core_news_sm')
        else:
            raise ValueError(f"Unsupported language {lang}")
        return self[lang]

SPACY_PIPES = __LazySpacyDict()

def extract_cn_baseword(uri: str):
    if uri.startswith('/'):
        return uri.split('/')[3]
    return "_".join(uri.split())

def pretokenize_cn_uri(uri: str):
    if uri.startswith('/'):
        return uri.split('/')[3].split('_')
    return uri.split()

# This function is called a lot. More efficient probably to not re-make the set every time
__tags = {"NOUN", "PNOUN", "ADJ", "ADV"}
def extract_lemmas(pipeline, sentence: str):
    global __tags
    return [t.lemma_ for t in pipeline(sentence) if t.pos_ in __tags]
    
