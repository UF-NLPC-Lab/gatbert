# STL
from __future__ import annotations
# 3rd Party
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
# Local
from typing import Dict, List, Tuple, Literal
from .data import make_encoder, make_collate_fn, MapDataset, parse_ez_stance
from .constants import DEFAULT_MODEL

class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 pretrained_model: str = DEFAULT_MODEL
                ):
        super().__init__()
        self.save_hyperparameters()

        tokenizer_model = AutoTokenizer.from_pretrained(self.hparams.pretrained_model, use_fast=True)
        # Protected variables
        self._encoder = make_encoder(tokenizer_model)
        self._collate_fn = make_collate_fn(tokenizer_model)

    # Protected Methods
    def _make_train_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._collate_fn)
    def _make_val_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._collate_fn)
    def _make_test_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._collate_fn)


class ByTargetDataModule(StanceDataModule):
    """
    Data module that partitions training, validation, and test sets based on the stance target.
    Should only be used with datasets with a small set of targets (e.g., SemEval2016-Task6)
    """
    def __init__(self,
                i: List[str],
                train_targets: List[str],
                val_targets: List[str],
                test_targets: List[str],
                *parent_args,
                **parent_kwargs
        ):
        super().__init__(*parent_args, **parent_kwargs)
        self.save_hyperparameters()

        self.__parse_fn = None
        raise ValueError("Parse function for SemEval not yet implemented")

        self.__data: MapDataset = None
        self.__train_ds: MapDataset = None
        self.__val_ds: MapDataset = None
        self.__test_ds: MapDataset = None

    def prepare_data(self):
        self.__data = MapDataset(map(self._encoder, self.__parse_fn(self.hparams.i)))

    def setup(self, stage):
        if stage == "fit" or stage is None:
            assert self.hparams.train_targets
            train_samples = []
            val_samples = []
            train_targs = set(self.hparams.train_targets)
            val_targs = set(self.hparams.val_targets)
            for sample in self.__data:
                if sample['target'] in train_targs:
                    train_samples.append(sample)
                elif sample['target'] in val_targs:
                    val_samples.append(sample)
            train_ds = MapDataset(train_samples)
            val_ds = MapDataset(val_samples)
            self.__train_ds = train_ds
            self.__val_ds = val_ds.map(self.remove_enhanced)
        if stage == "test" or stage is None:
            assert self.hparams.test_targets
            test_targs = set(self.hparams.test_targets)
            self.__test_ds = MapDataset(list(filter(lambda s: s['target'] in test_targs, self.__data))).map(self.remove_enhanced)

    def train_dataloader(self):
        return self._make_train_loader(self.__train_ds)
    def val_dataloader(self):
        return self._make_val_loader(self.__val_ds)
    def test_dataloader(self):
        return self._make_test_loader(self.__test_ds)

class RandomSplitDataModule(StanceDataModule):
    """
    Concatenates a collection of one or more files into one dataset,
    and randomly divides them among training, validation, and test.

    Useful for debugging, if you want to take the training partition of an existing
    dataset, and further partition it.
    """

    def __init__(self,
                partitions: Dict[str, Tuple[float, float, float]],
                corpus: Literal['ezstance', 'semeval', 'vast'],
                *parent_args,
                **parent_kwargs
        ):
        """
        
        Partitions is a mapping file_name->(train allocation, val allocation, test allocation).
        That is, for each data file, you specify how many of its samples go to training, validation, and test.
        """
        super().__init__(*parent_args, **parent_kwargs)
        self.save_hyperparameters()

        if corpus == "ezstance":
            self.__parse_fn = parse_ez_stance
        elif corpus == "semeval":
            raise ValueError("'semeval' parser not yet defined")
        else:
            raise ValueError("'vast' parser not yet defined")

        self.__data: Dict[str, MapDataset] = {}
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None

    def prepare_data(self):
        for data_path in self.hparams.partitions:
            self.__data[data_path] = MapDataset(map(self._encoder, self.__parse_fn(data_path)))

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