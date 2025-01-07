# STL
from __future__ import annotations
import csv
from typing import Optional, Dict, Any, Generator, List
import dataclasses
import pdb
# 3rd Party
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import PreTokenizer
# Local
from .constants import Stance, NodeType, DummyRelationType
from .graph import make_fake_kb_links


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

def parse_ez_stance(csv_path) -> Generator[Sample, None, None]:
    strstance2 = {"FAVOR": Stance.FAVOR, "AGAINST": Stance.AGAINST, "NONE": Stance.NONE}
    with open(csv_path, 'r', encoding='latin-1') as r:
        yield from map(lambda row: Sample(row['Text'], row['Target 1'], strstance2[row['Stance 1']]), csv.DictReader(r))

def make_encoder(tokenizer: PreTrainedTokenizerFast, pretokenizer: Optional[PreTokenizer] = None, add_fake_edges=False):

    def encode_sample(sample: Sample):
        context = sample.context
        target = sample.target
        stance = sample.stance.value

        if pretokenizer:
            pre_context = [pair[0] for pair in pretokenizer.pre_tokenize_str(context)]
            pre_target = [pair[0] for pair in pretokenizer.pre_tokenize_str(target)]
            result = tokenizer(text=pre_target, text_pair=pre_context, is_split_into_words=True, return_tensors='pt')
        else:
            result = tokenizer(text=target, text_pair=context, return_tensors='pt')

        result = {k: torch.squeeze(v) for (k, v) in result.items()}
        n_text_nodes = len(result['input_ids'])

        edge_ids = []
        for head in range(n_text_nodes):
            edge_ids.append( (head, head, DummyRelationType.TOKEN_TOKEN.value, NodeType.TOKEN.value, NodeType.TOKEN.value) )
            for tail in range(head + 1, n_text_nodes):
                edge_ids.append( (head, tail, DummyRelationType.TOKEN_TOKEN.value, NodeType.TOKEN.value, NodeType.TOKEN.value) )
                edge_ids.append( (tail, head, DummyRelationType.TOKEN_TOKEN.value, NodeType.TOKEN.value, NodeType.TOKEN.value) )

        if add_fake_edges:
            kb_edges, result['kb_ids'] = make_fake_kb_links(n_text_nodes)
            edge_ids.extend(kb_edges)
        else:
            result['kb_ids'] = torch.tensor([], dtype=torch.int64)

        edge_ids.sort()
        sparse_ids = torch.tensor(edge_ids).transpose(1, 0)
        result['edge_indices'] = sparse_ids
        result['stance'] = torch.tensor(stance)

        return result

    return encode_sample

def make_collate_fn(tokenizer):
    token_padding = tokenizer.pad_token_id
    type_padding = tokenizer.pad_token_type_id
    def collate_fn(samples: List[Dict[str, Any]]):
        batched = {}
        batched['input_ids'] = torch.nn.utils.rnn.pad_sequence([s['input_ids'] for s in samples], batch_first=True, padding_value=token_padding)
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