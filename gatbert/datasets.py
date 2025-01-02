# STL
from __future__ import annotations
import csv
from typing import Optional
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import PreTokenizer
# Local
from .constants import Stance

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
        self.__samples = samples
    def __getitem__(self, key):
        return self.__samples[key]
    def __len__(self):
        return len(self.__samples)

def parse_ez_stance(csv_path) -> MapDataset:
    strstance2enum = {
        "FAVOR": Stance.FAVOR,
        "AGAINST": Stance.AGAINST,
        "NONE": Stance.NONE
    }
    with open(csv_path, 'r', encoding='latin-1') as r:
        return MapDataset([
            {"context": row['Text'], "target": row["Target 1"], "stance": strstance2enum[row['Stance 1']] }
            for row in csv.DictReader(r)
        ])

def make_encoder(tokenizer: PreTrainedTokenizerFast, pretokenizer: Optional[PreTokenizer] = None):
    pass

def encode_dataset(ds: MapDataset):
    pass

class EZStanceDataset(RawDataset):
    def __init__(self, csv_path: str):
        super().__init__()

class EncodedDataset(torch.utils.data.Dataset):
    pass