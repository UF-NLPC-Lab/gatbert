import pathlib
from typing import List
import functools

from .parse import CorpusType, CORPUS_PARSERS
from .transforms import Transform

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
