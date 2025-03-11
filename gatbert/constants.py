import enum
import dataclasses
from typing import Literal
import re


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

DEFAULT_BATCH_SIZE = 4

DEFAULT_MAX_DEGREE = 50

DEFAULT_PG_ARGS = "dbname='conceptnet5' host='127.0.0.1'"
"""
Default connection arguments for postgres connections
"""

CN_URI_PATT = re.compile(r'/c/en/([^/]+)')

@dataclasses.dataclass
class Relation:
    orig_id: int
    internal_id: int
    name: str
    directed: bool

# (id, uri, is_directed) 
__RAW_RELATIONS = [
    ( 0, "/r/Antonym",                   False),
    ( 1, "/r/AtLocation",                True) ,
    ( 2, "/r/CapableOf",                 True) ,
    ( 3, "/r/Causes",                    True) ,
    ( 4, "/r/CausesDesire",              True) ,
    ( 5, "/r/CreatedBy",                 True) ,
    ( 6, "/r/DefinedAs",                 True) ,
    ( 7, "/r/DerivedFrom",               True) ,
    ( 8, "/r/Desires",                   True) ,
    ( 9, "/r/DistinctFrom",              False),
    (10, "/r/Entails",                   True) ,
    (11, "/r/EtymologicallyDerivedFrom", True) ,
    (12, "/r/EtymologicallyRelatedTo",   False),
    (13, "/r/ExternalURL",               True) ,
    (14, "/r/FormOf",                    True) ,
    (15, "/r/HasA",                      True) ,
    (16, "/r/HasContext",                True) ,
    (17, "/r/HasFirstSubevent",          True) ,
    (18, "/r/HasLastSubevent",           True) ,
    (19, "/r/HasPrerequisite",           True) ,
    (20, "/r/HasProperty",               True) ,
    (21, "/r/HasSubevent",               True) ,
    (22, "/r/InstanceOf",                True) ,
    (23, "/r/IsA",                       True) ,
    (24, "/r/LocatedNear",               True) ,
    (25, "/r/MadeOf",                    True) ,
    (26, "/r/MannerOf",                  True) ,
    (27, "/r/MotivatedByGoal",           True) ,
    (28, "/r/NotCapableOf",              True) ,
    (29, "/r/NotDesires",                True) ,
    (30, "/r/NotHasProperty",            True) ,
    (31, "/r/NotUsedFor",                True) ,
    (32, "/r/ObstructedBy",              True) ,
    (33, "/r/PartOf",                    True) ,
    (34, "/r/ReceivesAction",            True) ,
    (35, "/r/RelatedTo",                 False),
    (36, "/r/SimilarTo",                 False),
    (37, "/r/SymbolOf",                  True) ,
    (38, "/r/Synonym",                   False),
    (39, "/r/UsedFor",                   True) ,
    (40, "/r/dbpedia/capital",           True) ,
    (41, "/r/dbpedia/field",             True) ,
    (42, "/r/dbpedia/genre",             True) ,
    (43, "/r/dbpedia/genus",             True) ,
    (44, "/r/dbpedia/influencedBy",      True) ,
    (45, "/r/dbpedia/knownFor",          True) ,
    (46, "/r/dbpedia/language",          True) ,
    (47, "/r/dbpedia/leader",            True) ,
    (48, "/r/dbpedia/occupation",        True) ,
    (49, "/r/dbpedia/product",           True) ,
]
"""
A raw copy of the ConceptNet relations, for convenience.
"""

PADDING_RELATION_ID = 0
TOKEN_TO_TOKEN_RELATION_ID = 1
TOKEN_TO_KB_RELATION_ID = 2

CN_FORWARD_RELATIONS = {
        orig_id: Relation(orig_id, internal_id, name, directed) 
        for (internal_id, (orig_id, name, directed)) in enumerate(__RAW_RELATIONS, start=3)
}

CN_REV_RELATIONS = {
        orig_id: Relation(orig_id, internal_id, f"/reverse/{name}", directed) 
        for (internal_id, (orig_id, name, directed)) in enumerate(filter(lambda r: r[2], __RAW_RELATIONS), start=len(CN_FORWARD_RELATIONS)+3)
}

NUM_CN_RELATIONS = 3 + len(CN_FORWARD_RELATIONS) + len(CN_REV_RELATIONS)