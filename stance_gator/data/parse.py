from typing import Generator, Iterable, Callable, Literal, Dict
import argparse
import pathlib
import os
import json
import csv

from .stance import TriStance, BiStance
from .sample import Sample

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

def parse_standard(tsv_path) -> Generator[Sample, None, None]:
    stance_types = dict()
    with open(tsv_path, 'r') as r:
        reader = csv.DictReader(r, delimiter='\t')
        for row in reader:
            stance_type_name = row["StanceType"]
            if stance_type_name not in stance_types:
                stance_types[stance_type_name] = eval(stance_type_name)
            stance_type = stance_types[stance_type_name]
            yield Sample(context=row['Context'],
                         target=row['Target'],
                         stance=stance_type(int(row['Stance'])),
                         lang=row['Lang'] if row['Lang'] else None,
                         is_split_into_words=False)

def write_standard(out_path: os.PathLike, samples: Iterable[Sample]):
    with open(out_path, 'w') as w:
        writer = csv.writer(w, delimiter='\t')
        writer.writerow(["Target", "Context", "Stance", "StanceType", "Lang"])
        for sample in samples:
            writer.writerow([
                sample.target,
                sample.context,
                sample.stance,
                type(sample.stance).__name__,
                sample.lang]
            )

CorpusType = Literal['ezstance', 'semeval', 'vast', 'xstance', 'standard']

StanceParser = Callable[[os.PathLike], Generator[Sample, None, None]]
"""
Function taking a file path and returning a generator of samples
"""

CORPUS_PARSERS: Dict[CorpusType, StanceParser] = {
    "ezstance": parse_ez_stance,
    "vast": parse_vast,
    "semeval": parse_semeval,
    "xstance": parse_xstance,
    "standard": parse_standard
}

def add_corpus_args(parser: argparse.ArgumentParser):
    for name in CORPUS_PARSERS:
        parser.add_argument(f"--{name}", type=pathlib.Path, metavar="data.(csv|jsonl|tsv)")

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

