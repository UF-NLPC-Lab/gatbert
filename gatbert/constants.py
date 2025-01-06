import enum

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

# TODO: Should I make 
@enum.unique
class NodeType(enum.Enum):
    PADDING = 0
    TOKEN = 1
    KB = 2

@enum.unique
class DummyRelationType(enum.Enum):
    """
    Fake relations to use when debugging
    """
    PADDING = 0
    TOKEN_TOKEN = 1
    TOKEN_KB = 2
    KB_TOKEN = 3
    KB_KB = 4

ENCODED_FIELDS = {
    "stance",
    "input_ids",
    "position_ids",
}

DEFAULT_MODEL = "textattack/roberta-base-MNLI"
"""
See https://huggingface.co/textattack/roberta-base-MNLI
"""

NODE_PAD_ID = 0
"""
Padding ID to use when batches of nodes to equal sizes.
"""

DEFAULT_BATCH_SIZE = 4

NUM_FAKE_NODES = 1000
"""
Used for dummy routines testing graph logic
"""