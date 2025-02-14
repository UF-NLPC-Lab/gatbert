from __future__ import annotations
from typing import List, Generator, Optional
# 3rd party
# Local
from .data import parse_graph_tsv, parse_ez_stance, parse_semeval, parse_vast
from .graph_sample import GraphSample
from .types import CorpusType, TensorDict, Transform
from .encoder import Encoder

def map_func_gen(f, func):
    def mapped(*args, **kwargs):
        return map(f, func(*args, **kwargs))
    return mapped

class Preprocessor:
    def __init__(self,
                 corpus_type: CorpusType,
                 encoder: Encoder,
                 transforms: Optional[List[Transform]] = None):
        self.__encoder = encoder

        if corpus_type == 'graph':
            parse_fn = parse_graph_tsv
        elif corpus_type == 'ezstance':
            parse_fn = parse_ez_stance
        elif corpus_type == 'semeval':
            parse_fn = parse_semeval
        elif corpus_type == 'vast':
            parse_fn = parse_vast
        else:
            raise ValueError(f"Invalid corpus_type {corpus_type}")

        if transforms:
            transform_map = {
                'rm_external': self.rm_external
            }
            for t in transforms:
                parse_fn = map_func_gen(transform_map[t], parse_fn)
        parse_fn = map_func_gen(encoder.encode, parse_fn)
        self.__parse_fn = parse_fn


    def rm_external(self, sample: GraphSample) -> GraphSample:
        assert isinstance(sample, GraphSample), "'rm_external' only compatible with GraphSample"
        return sample.strip_external()

    def parse_file(self, path) -> Generator[TensorDict, None, None]:
        yield from self.__parse_fn(path)

    def collate(self, samples: List[TensorDict]) -> TensorDict:
        return self.__encoder.collate(samples)