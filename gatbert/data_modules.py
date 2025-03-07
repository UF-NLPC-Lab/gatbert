# STL
from __future__ import annotations
# 3rd Party
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
# Local
from typing import Dict, Tuple, Optional, List
from .data import MapDataset, parse_ez_stance, parse_graph_tsv, parse_semeval, parse_vast
from .constants import DEFAULT_BATCH_SIZE
from .stance_classifier import *
from .types import CorpusType, Transform
from .utils import map_func_gen

class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 corpus_type: CorpusType,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 transforms: Optional[List[Transform]] = None
                ):
        super().__init__()
        self.save_hyperparameters()

        if corpus_type == 'graph':
            parse_fn = parse_graph_tsv
        elif corpus_type == 'ezstance':
            parse_fn = parse_ez_stance
        elif corpus_type == 'semeval':
            parse_fn = parse_semeval
        elif corpus_type == 'vast':
            parse_fn = parse_vast
        else:
            raise ValueError(f"Invalid corpus_type {corpus_type}")
        if transforms:
            transform_map = {
                'rm_external': lambda s: s.strip_external() if isinstance(s, GraphSample) else s
            }
            for t in transforms:
                if t in transform_map:
                    parse_fn = map_func_gen(transform_map[t], parse_fn)

        self.__raw_parse_fn = parse_fn
        self.__parse_fn = None
        # Has to be set explicitly (see fit_and_test.py for an example)
        self.encoder: Encoder = None

    @property
    def _parse_fn(self):
        if not self.__parse_fn:
            if not self.encoder:
                raise ValueError("Encoder not set")
            self.__parse_fn = map_func_gen(self.encoder.encode, self.__raw_parse_fn)
        return self.__parse_fn

    # Protected Methods
    def _make_train_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self.encoder.collate)
    def _make_val_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self.encoder.collate)
    def _make_test_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self.encoder.collate)


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

        self.__data: Dict[str, MapDataset] = {}
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None

    def prepare_data(self):
        for data_path in self.hparams.partitions:
            self.__data[data_path] = MapDataset(self._parse_fn(data_path))

    def setup(self, stage):
        train_dses = []
        val_dses = []
        test_dses = []
        for (data_prefix, (train_frac, val_frac, test_frac)) in self.hparams.partitions.items():
            train_ds, val_ds, test_ds = \
                random_split(self.__data[data_prefix], [train_frac, val_frac, test_frac])
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