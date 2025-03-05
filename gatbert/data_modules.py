# STL
from __future__ import annotations
# 3rd Party
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
# Local
from typing import Dict, Tuple, Optional, List
from .data import MapDataset
from .preprocessor import Preprocessor
from .constants import DEFAULT_MODEL, DEFAULT_BATCH_SIZE
from .stance_classifier import *
from .types import CorpusType, Transform

class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 corpus_type: CorpusType,
                 classifier: type[StanceClassifier] = TextClassifier,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 tokenizer: str = DEFAULT_MODEL,
                 transforms: Optional[List[Transform]] = None,
                 graph: Optional[str] = None
                ):
        super().__init__()
        self.save_hyperparameters()

        tokenizer_model = AutoTokenizer.from_pretrained(self.hparams.tokenizer, use_fast=True)
        encoder = classifier.get_encoder(tokenizer_model, transforms, graph)
        # Protected variables
        self._preprocessor = Preprocessor(
            self.hparams.corpus_type, encoder, transforms
        )

    # Protected Methods
    def _make_train_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._preprocessor.collate)
    def _make_val_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._preprocessor.collate)
    def _make_test_loader(self, dataset: Dataset):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=self._preprocessor.collate)


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
            self.__data[data_path] = MapDataset(self._preprocessor.parse_file(data_path))

