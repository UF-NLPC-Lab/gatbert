# STL
from __future__ import annotations
import copy
# 3rd Party
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torch
import lightning as L
from tqdm import tqdm
# Local
from typing import Dict, Tuple, List
import random
from .sample import Sample
from .prompts import PROMPT_MAP, HYP_MAP
from .encoder import Encoder
from .data import MapDataset, CORPUS_PARSERS, CorpusType
from .constants import DEFAULT_BATCH_SIZE, BiStance
from .types import TensorDict
from .utils import map_func_gen

class StanceDataModule(L.LightningDataModule):
    def __init__(self,
                 corpus_type: CorpusType,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 prompt: bool = False
                ):
        super().__init__()
        self.save_hyperparameters()

        if corpus_type not in CORPUS_PARSERS:
            raise ValueError(f"Invalid corpus_type {corpus_type}")
        parse_fn = CORPUS_PARSERS[corpus_type]
        if prompt:
            def wrap_prompt(sample: Sample):
                copied = copy.deepcopy(sample)
                lang = sample.lang or 'en'
                assert not sample.is_split_into_words

                if lang == 'fr':
                    prompt = 'Intervieweur: "{target}" Politicien: "{context}" '
                else:
                    assert lang == 'de'
                    prompt = 'Interviewer: "{target}" Politiker: "{context}"'
                copied.context = prompt.format(target=sample.target, context=sample.context)

                if random.uniform(0, 1) > 0.5:
                    copied.stance = BiStance((sample.stance + 1) % 2)
                    if lang == 'de':
                        sample.target = "Der Politiker ist anderer Meinung als der Interviewer."
                    else:
                        sample.target = "Le politicien n'est pas d'accord avec l'intervieweur."
                else:
                    if lang == 'fr':
                        sample.target = "Le politicien était d'accord avec l'intervieweur."
                    else:
                        sample.target = "Der Politiker stimmt dem Interviewer zu."
                return copied
            parse_fn = map_func_gen(wrap_prompt, parse_fn)

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

    def setup(self, stage):
        for data_path in self.hparams.partitions:
            parse_iter = tqdm(self._parse_fn(data_path), desc=f"Parsing {data_path}")
            self._data[data_path] = MapDataset(parse_iter)

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