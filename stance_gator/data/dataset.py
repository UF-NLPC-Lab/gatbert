# STL
from __future__ import annotations
# 3rd Party
import torch
# Local

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
