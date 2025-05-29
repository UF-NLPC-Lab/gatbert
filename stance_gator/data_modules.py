# STL
from __future__ import annotations
# 3rd Party
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torch
import lightning as L
from tqdm import tqdm
import pathlib
# Local
from typing import Dict, Tuple, List, Tuple
from .encoder import Encoder
from .data import MapDataset, CORPUS_PARSERS, CorpusType
from .constants import DEFAULT_BATCH_SIZE
from .types import TensorDict
from .utils import map_func_gen

class StanceCorpus:
    def __init__(self,
                 path: pathlib.Path,
                 corpus_type: CorpusType,
                 data_ratio:  Tuple[float, float, float]):
        if corpus_type not in CORPUS_PARSERS:
            raise ValueError(f"Invalid corpus_type {corpus_type}")
        self.parse_fn = CORPUS_PARSERS[corpus_type]
        self.data_ratio = data_ratio
        self.path = path


class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int = DEFAULT_BATCH_SIZE
                ):
        super().__init__()
        # Has to be set explicitly (see fit_and_test.py for an example)
        self.encoder: Encoder = None
        self.batch_size = batch_size

    @property
    def _collate_fn(self):
        return self.encoder.collate

    # Protected Methods
    def _make_train_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn, shuffle=True)
    def _make_val_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)
    def _make_test_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)


class RandomSplitDataModule(StanceDataModule):
    """
    Concatenates a collection of one or more files into one dataset,
    and randomly divides them among training, validation, and test.

    Useful for debugging, if you want to take the training partition of an existing
    dataset, and further partition it.
    """

    def __init__(self,
                corpora: List[StanceCorpus],
                *parent_args,
                **parent_kwargs
        ):
        """
        
        Partitions is a mapping file_name->(train allocation, val allocation, test allocation).
        That is, for each data file, you specify how many of its samples go to training, validation, and test.
        """
        super().__init__(*parent_args, **parent_kwargs)
        self.save_hyperparameters()

        self._data: Dict[str, MapDataset] = {}
        self._corpora = corpora
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None

    def setup(self, stage):
        if self.__train_ds and self.__val_ds and self.__test_ds:
            return

        train_dses = []
        val_dses = []
        test_dses = []
        for corpus in self._corpora:
            parse_iter = tqdm(corpus.parse_fn(corpus.path), desc=f"Parsing {corpus.path}")
            encoded = MapDataset(map(self.encoder.encode, parse_iter))
            train_ds, val_ds, test_ds = \
                random_split(encoded, corpus.data_ratio)
            train_dses.append(train_ds)
            val_dses.append(val_ds)
            test_dses.append(test_ds)
        self.__train_ds = ConcatDataset(train_dses)
        self.__val_ds = ConcatDataset(val_dses)
        self.__test_ds = ConcatDataset(test_dses)

    def train_dataloader(self):
        return self._make_train_loader(self.__train_ds)
    def val_dataloader(self):
        return self._make_val_loader(self.__val_ds)
    def test_dataloader(self):
        return self._make_test_loader(self.__test_ds)

class GraphRandomSplitDataModule(RandomSplitDataModule):
    def __init__(self,
                graph_data: Dict[str, str],
                partitions: Dict[str, Tuple[float, float, float]],
                *parent_args,
                **parent_kwargs
        ):
        super().__init__(*parent_args, partitions=partitions, **parent_kwargs)
        graph_keys = set(graph_data)
        for k in partitions.keys():
            assert k in graph_keys, f"File {k} has no matching graph data .npy file"
        self.graph_data = graph_data

        self.__parse_fn = None
        self.__collate_fn = None

    @property
    def _parse_fn(self):
        if not self.__parse_fn:
            parent_parse = super()._parse_fn
            def wrapper(corpus_path):
                graph_data = np.load(self.graph_data[corpus_path], allow_pickle=True)
                for text_encoding, graph_encoding in zip(parent_parse(corpus_path), graph_data):
                    yield {**text_encoding, 'graph_embeds': torch.tensor(graph_encoding, dtype=torch.float32)}
            self.__parse_fn = wrapper
        return self.__parse_fn

    @property
    def _collate_fn(self):
        if not self.__collate_fn:
            parent_collate = super()._collate_fn
            def wrapper(samples: List[TensorDict]) -> TensorDict:
                # graph_embeds = torch.stack([s.pop('graph_embeds') for s in samples], dim=0)
                graph_embeds = torch.stack([s['graph_embeds'] for s in samples], dim=0)
                collated = parent_collate(samples)
                collated['graph_embeds'] = graph_embeds
                return collated
            return wrapper
        return self.__collate_fn