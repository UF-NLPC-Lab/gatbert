# STL
from __future__ import annotations
# 3rd Party
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
from tqdm import tqdm
from typing import Tuple, List, Tuple
# Local
from .encoder import Encoder
from .dataset import MapDataset
from .corpus import StanceCorpus
from ..constants import DEFAULT_BATCH_SIZE



class VizDataModule(L.LightningDataModule):
    def __init__(self, corpus: StanceCorpus, batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.save_hyperparameters()
        self.corpus = corpus
        self.encoder: Encoder = None
        self.batch_size = batch_size

        self.__ds: Dataset = None
    def setup(self, stage):
        if self.__ds is not None:
            return
        corpus = self.corpus
        parse_iter = tqdm(corpus.parse_fn(corpus.path), desc=f'Parsing {corpus.path}')

        encodings = []

        for sample in parse_iter:
            encoding = self.encoder.encode(sample)
            encodings.append(encoding)
        self.__ds = MapDataset(encodings)

    def predict_dataloader(self):
        return DataLoader(self.__ds,
                          batch_size=self.batch_size,
                          collate_fn=self.encoder.collate,
                          shuffle=False)
    def test_dataloader(self):
        return self.predict_dataloader()

class SplitDataModule(L.LightningDataModule):
    def __init__(self,
                 corpora: List[StanceCorpus],
                 ratios: List[Tuple[float, float, float]],
                 batch_size: int = DEFAULT_BATCH_SIZE
                ):
        super().__init__()
        # Has to be set explicitly (see cli.py for an example)
        self.encoder: Encoder = None
        self.batch_size = batch_size
        self._corpora = corpora
        self._ratios = ratios
        assert len(self._corpora) == len(self._ratios)
        self.__train_ds: Dataset = None
        self.__val_ds: Dataset = None
        self.__test_ds: Dataset = None


    def setup(self, stage):
        if self.__train_ds and self.__val_ds and self.__test_ds:
            return

        train_dses = []
        val_dses = []
        test_dses = []
        for corpus, data_ratio in zip(self._corpora, self._ratios):
            parse_iter = tqdm(corpus.parse_fn(corpus.path), desc=f"Parsing {corpus.path}")
            encoded = MapDataset(map(self.encoder.encode, parse_iter))
            train_ds, val_ds, test_ds = \
                random_split(encoded, data_ratio)
            train_dses.append(train_ds)
            val_dses.append(val_ds)
            test_dses.append(test_ds)
        self.__train_ds = ConcatDataset(train_dses)
        self.__val_ds = ConcatDataset(val_dses)
        self.__test_ds = ConcatDataset(test_dses)

    def train_dataloader(self):
        return DataLoader(self.__train_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.__val_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate)
    def test_dataloader(self):
        return DataLoader(self.__test_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate)