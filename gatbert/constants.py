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

ENCODED_FIELDS = {
    "stance",
    "input_ids",
    "position_ids",
}

DEFAULT_MODEL = "textattack/roberta-base-MNLI"
"""
See https://huggingface.co/textattack/roberta-base-MNLI
"""

DEFAULT_BATCH_SIZE = 4
