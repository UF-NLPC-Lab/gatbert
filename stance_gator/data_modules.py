# STL
from __future__ import annotations
# 3rd Party
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torch
import lightning as L
# Local
from typing import Dict, Tuple, Optional, List
from .encoder import Encoder
from .data import MapDataset, parse_ez_stance, parse_semeval, parse_vast
from .constants import DEFAULT_BATCH_SIZE
from .types import CorpusType, TensorDict
from .utils import map_func_gen

class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 corpus_type: CorpusType,
                 batch_size: int = DEFAULT_BATCH_SIZE
                ):
        super().__init__()
        self.save_hyperparameters()

        if corpus_type == 'ezstance':
            parse_fn = parse_ez_stance
        elif corpus_type == 'semeval':
            parse_fn = parse_semeval
        elif corpus_type == 'vast':
            parse_fn = parse_vast
        else:
            raise ValueError(f"Invalid corpus_type {corpus_type}")

        self.__raw_parse_fn = parse_fn
        self.__parse_fn = None
        # Has to be set explicitly (see fit_and_test.py for an example)
        self.encoder: Encoder = None
        self.batch_size = batch_size

    @property
    def _parse_fn(self):
        if not self.__parse_fn:
            if not self.encoder:
                raise ValueError("Encoder not set")
            self.__parse_fn = map_func_gen(self.encoder.encode, self.__raw_parse_fn)
        return self.__parse_fn

    @property
    def _collate_fn(self):
        return self.encoder.collate

    # Protected Methods
    def _make_train_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)
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
                partitions: Dict[str, Tuple[float, float, float]],
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
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None

    def prepare_data(self):
        for data_path in self.hparams.partitions:
            self._data[data_path] = MapDataset(self._parse_fn(data_path))

    def setup(self, stage):
        train_dses = []
        val_dses = []
        test_dses = []
        for (data_prefix, (train_frac, val_frac, test_frac)) in self.hparams.partitions.items():
            train_ds, val_ds, test_ds = \
                random_split(self._data[data_prefix], [train_frac, val_frac, test_frac])
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
        assert set(partitions.keys()) == set(graph_data.keys())
        self.graph_data = graph_data

        self.__parse_fn = None
        self.__collate_fn = None

    @property
    def _parse_fn(self):
        if not self.__parse_fn:
            parent_parse = super()._parse_fn
            def wrapper(corpus_path):
                graph_data = np.load(self.graph_data[corpus_path], allow_pickle=False)
                for text_encoding, graph_encoding in zip(parent_parse(corpus_path), graph_data):
                    yield {**text_encoding, 'graph_embeds': torch.tensor(graph_encoding)}
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