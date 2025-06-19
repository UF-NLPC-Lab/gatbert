# STL
from __future__ import annotations
import copy
import functools
# 3rd Party
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as L
from tqdm import tqdm
import pathlib
from typing import Tuple, List, Tuple
# Local
from .sample import Sample
from .encoder import Encoder
from .data import MapDataset, CORPUS_PARSERS, CorpusType, SPACY_PIPES
from .constants import DEFAULT_BATCH_SIZE, BaseStance, BiStance, TriStance
from .cn import load_syn_map

class Transform:
    def __call__(self, sample: Sample) -> Sample:
        raise NotImplementedError

class LabelTransform(Transform):
    def __init__(self, target_type: type[BaseStance]):
        self.target_type = target_type
    def __call__(self, sample: Sample) -> Sample:
        if isinstance(sample.stance, self.target_type):
            return sample
        if not isinstance(sample.stance, BaseStance):
            raise ValueError(f"Invalid stance type {type(sample.stance)}")
        stance_val = sample.stance
        if self.target_type is BiStance:
            assert isinstance(stance_val, TriStance)
            if sample == TriStance.neutral:
                raise ValueError(f"Cannot convert a neutral stance to BiStance")
            sample.stance = self.target_type(stance_val.value)
        else:
            assert self.target_type is TriStance
            assert isinstance(stance_val, BiStance)
            sample.stance = self.target_type(stance_val.value)
        return sample

class ReweightTransform(Transform):
    def __init__(self, new_weight: float):
        self.new_weight = new_weight
    def __call__(self, sample: Sample) -> Sample:
        sample.weight = self.new_weight
        return sample

class SynTransform(Transform):
    def __init__(self, syn_path: pathlib.Path):
        self.adj = load_syn_map(syn_path)
        self.en_pipe = SPACY_PIPES['en']

    @functools.lru_cache
    def __syn_lookup(self, target: str, lang: str):
        spacy_doc = self.en_pipe(target.lower())
        rval = []
        for tok in spacy_doc:
            lemma = tok.lemma_
            rval.append(self.adj.get(lang, {}).get(lemma, lemma))
        return " ".join(rval)

    def __call__(self, sample: Sample) -> Sample:
        lang = sample.lang or 'en'
        if lang == 'en':
            return sample
        if sample.is_split_into_words:
            sample.context = " ".join(sample.context)
            sample.target = " ".join(sample.target)
            sample.is_split_into_words = False
        sample.target = self.__syn_lookup(sample.target, lang)
        return sample

class StanceCorpus:
    def __init__(self,
                 path: pathlib.Path,
                 corpus_type: CorpusType,
                 transforms: List[Transform] = []):
        if corpus_type not in CORPUS_PARSERS:
            raise ValueError(f"Invalid corpus_type {corpus_type}")
        self._parse_fn = CORPUS_PARSERS[corpus_type]
        self.path = path
        self.transforms = transforms

    def parse_fn(self, *args, **kwargs):
        for sample in self._parse_fn(*args, **kwargs):
            sample = functools.reduce(lambda s, f: f(s), self.transforms, sample)
            yield sample

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