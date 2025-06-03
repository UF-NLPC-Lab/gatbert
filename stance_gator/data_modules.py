# STL
from __future__ import annotations
# 3rd Party
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
from tqdm import tqdm
import pathlib
# Local
from typing import Any, Tuple, List, Tuple
from .encoder import Encoder
from .data import MapDataset, CORPUS_PARSERS, CorpusType
from .constants import DEFAULT_BATCH_SIZE
from .utils import batched

class StanceCorpus:
    def __init__(self,
                 path: pathlib.Path,
                 corpus_type: CorpusType):
        if corpus_type not in CORPUS_PARSERS:
            raise ValueError(f"Invalid corpus_type {corpus_type}")
        self.parse_fn = CORPUS_PARSERS[corpus_type]
        self.path = path

class SplitCorpus(StanceCorpus):
    def __init__(self,
                 path: pathlib.Path,
                 corpus_type: CorpusType,
                 data_ratio:  Tuple[float, float, float]):
        super().__init__(path=path, corpus_type=corpus_type)
        self.data_ratio = data_ratio


class VizDataModule(L.LightningDataModule):
    def __init__(self, corpus: StanceCorpus, batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.save_hyperparameters()
        self.corpus = corpus
        self.encoder: Encoder = None
        self.batch_size = batch_size

        self.__ds: Dataset = None
        self.__meta: List[Any] = None
    def setup(self, stage):
        if self.__ds is not None:
            return
        corpus = self.corpus
        parse_iter = tqdm(corpus.parse_fn(corpus.path), desc=f'Parsing {corpus.path}')

        encodings = []
        metadatas = []

        for sample in parse_iter:
            encoding, metadata = self.encoder.encode_with_meta(sample)
            encodings.append(encoding)
            metadatas.append(metadata)
        self.__ds = MapDataset(encodings)
        self.__meta = metadatas

    def predict_dataloader(self):
        return DataLoader(self.__ds,
                          batch_size=self.batch_size,
                          collate_fn=self.encoder.collate,
                          shuffle=False)
    def predict_metadata(self):
        return list(batched(self.__meta, self.batch_size))

class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 corpora: List[SplitCorpus],
                 batch_size: int = DEFAULT_BATCH_SIZE
                ):
        super().__init__()
        # Has to be set explicitly (see cli.py for an example)
        self.encoder: Encoder = None
        self.batch_size = batch_size
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
        return DataLoader(self.__train_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.__val_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate)
    def test_dataloader(self):
        return DataLoader(self.__test_ds, batch_size=self.batch_size, collate_fn=self.encoder.collate)