# STL
from __future__ import annotations
import argparse
import csv
from typing import Generator
# 3rd Party
import torch
import spacy
# Local
from .constants import TriStance
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
                      is_split_into_words=False,
                      domain=row['Domain'] if 'Domain' in row else None)
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

CORPUS_PARSERS = {
    "ezstance": parse_ez_stance,
    "vast": parse_vast,
    "semeval": parse_semeval
}

def add_corpus_args(parser: argparse.ArgumentParser):
    parser.add_argument("--vast", metavar="vast_train.csv")
    parser.add_argument("--ezstance", metavar="raw_train_all_onecol.csv")
def get_corpus_parser(args):
    assert sum([bool(args.vast), bool(args.ezstance)]) == 1, "Must provided one of --vast, --ezstance"
    if args.vast:
        return parse_vast(args.vast)
    return  parse_ez_stance(args.ezstance)

__spacy_pipeline = None
def get_en_pipeline():
    global __spacy_pipeline
    if __spacy_pipeline is None:
        __spacy_pipeline = spacy.load('en_core_web_sm')
    return __spacy_pipeline

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
    
