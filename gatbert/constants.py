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

@enum.unique
class Stance(enum.Enum):
    NONE = 0
    AGAINST = 1
    FAVOR = 2

@enum.unique
class NodeType(enum.Enum):
    TOKEN = 0
    KB = 1

ENCODED_FIELDS = {
    "stance",
    "input_ids",
    "position_ids",
}

DEFAULT_ATT_TYPE = 'edge_as_att'

# DEFAULT_MODEL = "textattack/roberta-base-MNLI"
DEFAULT_MODEL = "bert-base-uncased"
"""
See https://huggingface.co/textattack/roberta-base-MNLI
"""

NODE_PAD_ID = 0
"""
Padding ID to use when batches of nodes to equal sizes.
"""

MAX_KB_NODES = 128
"""
Maximum number of subwords to allow from external (not text) data.
"""

DEFAULT_BATCH_SIZE = 64

DEFAULT_MAX_DEGREE = 50

DEFAULT_PG_ARGS = "dbname='conceptnet5' host='127.0.0.1'"
"""
Default connection arguments for postgres connections
"""

CN_URI_PATT = re.compile(r'/c/en/([^/]+)')

OPTUNA_STUDY_NAME = "gatbert_transe"

PYKEEN_METRIC = "both.realistic.inverse_harmonic_mean_rank"

@enum.unique
class SpecialRelation(enum.Enum):
    TOKEN_TO_TOKEN = -1
    """
    Token-to-token relations (that is, what a regular transformer encodes already)
    """

    TOKEN_TO_KB = -2
    """
    Arbitrary relation between a token and a KB node (the 'bridge' between tokens and the KG)
    """

    KB_SIM = -3
    """
    BERT embeddings of these two KB nodes are similar
    """