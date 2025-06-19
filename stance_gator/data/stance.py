import enum
from typing import Dict, Literal

@enum.unique
class BaseStance(enum.IntEnum):

    @classmethod
    def label2id(cls):
        return {s.name:s for s in cls}

    @classmethod
    def id2label(cls):
        return {v:k for k,v in cls.label2id().items()}

@enum.unique
class TriStance(BaseStance):
    against = 0
    favor = 1
    neutral = 2

@enum.unique
class BiStance(BaseStance):
    against = 0
    favor = 1

StanceType = Literal['tri', 'bi']

STANCE_TYPE_MAP: Dict[StanceType, BaseStance] = {
    'tri': TriStance,
    'bi': BiStance
}

