import functools
import pathlib
from typing import Optional

from .stance import BaseStance, TriStance, BiStance, STANCE_TYPE_MAP, StanceType
from .sample import Sample
from .spacy_utils import SPACY_PIPES
from .cn import load_syn_map

class Transform:
    def __call__(self, sample: Sample) -> Sample:
        raise NotImplementedError

class ClassWeightTransform(Transform):
    def __init__(self,
                 favor: Optional[float] = None,
                 against: Optional[float] = None,
                 neutral: Optional[float] = None
                 ):
        self.favor = favor
        self.against = against
        self.neutral = neutral
    def __call__(self, sample: Sample) -> Sample:
        stance_name = sample.stance.name
        if stance_name == 'favor':
            if self.favor is not None:
                sample.weight = self.favor
        elif stance_name == 'against':
            if self.against is not None:
                sample.weight = self.against
        elif stance_name == 'neutral':
            if self.neutral is not None:
                sample.weight = self.neutral
        return sample

class LabelTransform(Transform):
    def __init__(self, target_type: StanceType):
        self.target_type = STANCE_TYPE_MAP[target_type]
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
