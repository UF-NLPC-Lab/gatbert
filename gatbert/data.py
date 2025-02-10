# STL
from __future__ import annotations
import csv
from typing import Dict, Any, Generator, List, Callable, Literal
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import BertPreTokenizer
# Local
from .constants import Stance
from .types import CorpusType, SampleType, TensorDict
from .sample import Sample, PretokenizedSample
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

def map_func_gen(f, func):
    def mapped(*args, **kwargs):
        return map(f, func(*args, **kwargs))
    return mapped

class Preprocessor:
    def __init__(self,
                 corpus_type: CorpusType,
                 sample_type: SampleType,
                 tokenizer: PreTrainedTokenizerFast):
        self.corpus_type = corpus_type
        self.sample_type = sample_type
        self.tokenizer = tokenizer

        if corpus_type == 'graph':
            parse_fn = parse_graph_tsv
            if sample_type == 'token':
                encoder = PretokenizedSample.Encoder(self.tokenizer)
                parse_fn = map_func_gen(lambda gs: encoder.encode(gs.to_sample()), parse_fn)
                collate_fn = encoder.collate
            elif sample_type == 'concat':
                encoder = GraphSample.ConcatEncoder(self.tokenizer)
                parse_fn = map_func_gen(encoder.encode, parse_fn)
                collate_fn = encoder.collate
            elif sample_type in {'graph', 'stripped_graph'}:
                if sample_type == 'stripped_graph':
                    parse_fn = map_func_gen(GraphSample.strip_external, parse_fn)
                gs_encoder = GraphSample.Encoder(self.tokenizer)
                parse_fn = map_func_gen(gs_encoder.encode, parse_fn)
                collate_fn = gs_encoder.collate
            else:
                raise ValueError(f"Invalid sample_type {sample_type}")
        else:
            assert sample_type == 'token'
            if corpus_type == 'ezstance':
                parse_fn = parse_ez_stance
            elif corpus_type == 'semeval':
                parse_fn = parse_semeval
            elif corpus_type == 'vast':
                parse_fn = parse_vast
            else:
                raise ValueError(f"Invalid corpus_type {corpus_type}")
            encoder = Sample.Encoder(self.tokenizer)
            parse_fn = map_func_gen(encoder.encode, parse_fn)
            collate_fn = encoder.collate

        self.__parse_fn = parse_fn
        self.__collate_fn = collate_fn

    def __encode_sample(self, sample: Sample) -> TensorDict:
        result = self.tokenizer(text=sample.target, text_pair=sample.context, return_tensors='pt')
        if 'token_type_ids' in result:
            result.pop('token_type_ids')
        result['stance'] = torch.tensor(sample.stance.value, device=result['input_ids'].device)
        return result

    def __encode_pretokenized(self, sample: PretokenizedSample) -> TensorDict:
        result = self.tokenizer(text=sample.target, text_pair=sample.context, is_split_into_words=True, return_tensors='pt')
        result['stance'] = torch.tensor(sample.stance.value, device=result['input_ids'].device)
        return result

    def __simple_collate(self, samples: List[TensorDict]) -> TensorDict:
        token_padding = self.tokenizer.pad_token_id
        type_padding = self.tokenizer.pad_token_type_id
        batched = {}
        batched['input_ids'] = torch.nn.utils.rnn.pad_sequence([s['input_ids'].squeeze() for s in samples], batch_first=True, padding_value=token_padding)
        if "token_type_ids" in samples[0]:
            batched['token_type_ids'] = torch.nn.utils.rnn.pad_sequence([s['token_type_ids'].squeeze() for s in samples], batch_first=True, padding_value=type_padding)
        batched['attention_mask'] = batched['input_ids'] != token_padding
        batched['stance'] = torch.stack([s['stance'] for s in samples], dim=0)
        return batched

    def parse_file(self, path) -> Generator[TensorDict, None, None]:
        yield from self.__parse_fn(path)

    def collate(self, samples: List[TensorDict]) -> TensorDict:
        return self.__collate_fn(samples)