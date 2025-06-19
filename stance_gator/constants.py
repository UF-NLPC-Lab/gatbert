import enum
import re

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


DEFAULT_MODEL = "bert-base-uncased"

DEFAULT_BATCH_SIZE = 64

CN_URI_PATT = re.compile(r'/c/en/([^/]+)')
