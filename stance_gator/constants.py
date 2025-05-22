import enum
import re
from typing import Literal, Dict

@enum.unique
class EzstanceDomains(enum.Enum):
    COVID = "covid19_domain"
    WORLD = "world_event_domain"
    EDU_CUL = "education_and_culture_domain"
    ENT_CON = "consumption_and_entertainment_domain"
    SPORTS = "sports_domain"
    RIGHTS = "rights_domain"
    ENV = "environmental_protection_domain"
    POL = "politic"


@enum.unique
class SemEvalTargets(enum.Enum):
    ABORTION = "Legalization of Abortion"
    ATHEISM = "Atheism"
    CLIMATE_CHANGE = "Climate Change is a Real Concern"
    DONALD_TRUMP = "Donald Trump"
    FEMINISM = "Feminist Movement"
    HILLARY_CLINTON = "Hillary Clinton"

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
    neutral = 0
    against = 1
    favor = 2

@enum.unique
class BiStance(BaseStance):
    against = 0
    favor = 1

StanceType = Literal['tri', 'bi']

STANCE_TYPE_MAP: Dict[StanceType, BaseStance] = {
    'tri': TriStance,
    'bi': BiStance
}


DEFAULT_MODEL = "bert-base-uncased"

DEFAULT_BATCH_SIZE = 64

CN_URI_PATT = re.compile(r'/c/en/([^/]+)')
