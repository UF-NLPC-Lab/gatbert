# STL
from __future__ import annotations
# 3rd Party
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
# Local
from typing import Dict, List, Tuple, Literal
from .data import make_file_parser, make_collate_fn, MapDataset
from .constants import DEFAULT_MODEL
from .types import CorpusType

class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 corpus: CorpusType,
                 pretrained_model: str = DEFAULT_MODEL
                ):
        super().__init__()
        self.save_hyperparameters()

        tokenizer_model = AutoTokenizer.from_pretrained(self.hparams.pretrained_model, use_fast=True)

        # Protected variables
        self._file_parser = make_file_parser(self.hparams.corpus, tokenizer_model)
        self._collate_fn = make_collate_fn(self.hparams.corpus, tokenizer_model)

    # Protected Methods
    def _make_train_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._collate_fn)
    def _make_val_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._collate_fn)
    def _make_test_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._collate_fn)


class RandomSplitDataModule(StanceDataModule):
    """
    Concatenates a collection of one or more files into one dataset,
    and randomly divides them among training, validation, and test.

    Useful for debugging, if you want to take the training partition of an existing
    dataset, and further partition it.
    """

    def __init__(self,
                partitions: Dict[str, Tuple[float, float, float]],
                corpus: CorpusType,
                *parent_args,
                **parent_kwargs
        ):
        """
        
        Partitions is a mapping file_name->(train allocation, val allocation, test allocation).
        That is, for each data file, you specify how many of its samples go to training, validation, and test.
        """
        super().__init__(*parent_args, corpus=corpus, **parent_kwargs)
        self.save_hyperparameters()

        self.__data: Dict[str, MapDataset] = {}
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None

    def prepare_data(self):
        for data_path in self.hparams.partitions:
            self.__data[data_path] = MapDataset(self._file_parser(data_path))

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