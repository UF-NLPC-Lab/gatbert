# STL
from __future__ import annotations
import csv
from typing import Dict, Any, Generator, List, Callable
import dataclasses
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import BertPreTokenizer
# Local
from .constants import Stance
from .types import CorpusType, TensorDict
from .graph_sample import GraphSample


class MapDataset(torch.utils.data.Dataset):
    """
    In-memory dataset for stance samples
    """

    @staticmethod
    def from_dataset(ds: torch.utils.data.Dataset):
        return MapDataset(list(ds))

    def filter(self, pred) -> MapDataset:
        return MapDataset(list(filter(pred, self.__samples)))
    
    def map(self, f) -> MapDataset:
        return MapDataset(list(map(f, self.__samples)))

    def __init__(self, samples):
        self.__samples = list(samples)
    def __getitem__(self, key):
        return self.__samples[key]
    def __len__(self):
        return len(self.__samples)

@dataclasses.dataclass
class Sample:
    context: str
    target: str
    stance: Stance

@dataclasses.dataclass
class PretokenizedSample:
    context: List[str]
    target: List[str]
    stance: Stance


def get_default_pretokenize() -> Callable[[Sample], PretokenizedSample]:
    pretok = BertPreTokenizer()
    def f(sample: Sample):
        return PretokenizedSample(
            context=[pair[0] for pair in pretok.pre_tokenize_str(sample.context)],
            target=[pair[0] for pair in pretok.pre_tokenize_str(sample.target)],
            stance=sample.stance
        )
    return f

def parse_graph_tsv(tsv_path) -> Generator[GraphSample, None, None]:
    with open(tsv_path, 'r') as r:
        yield from map(GraphSample.from_row, csv.reader(r, delimiter='\t'))

def parse_ez_stance(csv_path) -> Generator[Sample, None, None]:
    strstance2 = {"FAVOR": Stance.FAVOR, "AGAINST": Stance.AGAINST, "NONE": Stance.NONE}
    with open(csv_path, 'r', encoding='latin-1') as r:
        yield from map(lambda row: Sample(row['Text'], row['Target 1'], strstance2[row['Stance 1']]), csv.DictReader(r))

def parse_vast(csv_path) -> Generator[Sample, None, None]:
    raise NotImplementedError

def parse_semeval(annotations_path) -> Generator[Sample, None, None]:
    raise NotImplementedError

def make_file_parser(corpus_type: CorpusType, tokenizer: PreTrainedTokenizerFast) -> Callable[[str], Generator[TensorDict, None, None]]:
    """
    Returns:
        Function that takes a file path and returns a generator of samples
    """
    if corpus_type in {'graph', 'graph_token'}:
        def f(file_path: str):
            sample_gen = parse_graph_tsv(file_path)
            if corpus_type == 'graph_token':
                sample_gen = map(GraphSample.strip_external, sample_gen)
            sample_gen = map(lambda sample: sample.encode(tokenizer), sample_gen)
            yield from sample_gen
    else:
        if corpus_type == 'ezstance':
            parse_fn = parse_ez_stance
        elif corpus_type == 'vast':
            parse_fn = parse_vast
        elif corpus_type == 'semeval':
            parse_fn = parse_semeval
        else:
            raise ValueError(f"Invalid corpus_type {corpus_type}")
        def f(file_path: str):
            for sample in parse_fn(file_path):
                result = tokenizer(text=sample.target, text_pair=sample.context, return_tensors='pt')
                result['stance'] = torch.tensor(sample.stance.value, device=result['input_ids'].device)
                yield result
    return f

def make_collate_fn(corpus_type: CorpusType, tokenizer: PreTrainedTokenizerFast) -> Callable[[TensorDict], TensorDict]:
    if corpus_type in {'graph', 'graph_token'}:
        return GraphSample.collate

    token_padding = tokenizer.pad_token_id
    type_padding = tokenizer.pad_token_type_id
    def collate_fn(samples: List[Dict[str, Any]]):
        batched = {}
        batched['input_ids'] = torch.nn.utils.rnn.pad_sequence([s['input_ids'] for s in samples], batch_first=True, padding_value=token_padding)
        if "token_type_ids" in samples[0]:
            batched['token_type_ids'] = torch.nn.utils.rnn.pad_sequence([s['token_type_ids'] for s in samples], batch_first=True, padding_value=type_padding)
        batched['kb_ids'] = torch.nn.utils.rnn.pad_sequence([s['kb_ids'] for s in samples], batch_first=True, padding_value=0)

        batched['attention_mask'] = batched['input_ids'] != token_padding
        batched['stance'] = torch.stack([s['stance'] for s in samples], dim=0)

        batch_edges = []
        for (i, sample_edges) in enumerate(map(lambda s: s['edge_indices'], samples)):
            batch_edges.append(torch.concatenate([
                torch.full(size=(1, sample_edges.shape[1]), fill_value=i),
                sample_edges
            ]))
        batched['edge_indices'] = torch.concatenate(batch_edges, dim=-1)
        return batched
    return collate_fn