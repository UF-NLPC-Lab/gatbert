# STL
from __future__ import annotations
import csv
from typing import Generator
# 3rd Party
import torch
# Local
from .constants import Stance, EzstanceDomains
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
    strstance2 = {"FAVOR": Stance.FAVOR, "AGAINST": Stance.AGAINST, "NONE": Stance.NONE}
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
       "0": Stance.AGAINST,
       "1": Stance.FAVOR,
       "2": Stance.NONE
    }
    # TODO: Do we still want to use "post", or one of their preprocessed versions?
    with open(csv_path, 'r') as r:
        yield from map(lambda row: Sample(row['post'], row['topic_str'], strstance2enum[row['label']], is_split_into_words=False), csv.DictReader(r))

def parse_semeval(annotations_path) -> Generator[Sample, None, None]:
    raise NotImplementedError
