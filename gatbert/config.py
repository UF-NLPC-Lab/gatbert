# 3rd Party
from transformers import PretrainedConfig
# Local
from .types import AttentionType

class GatbertConfig:
    """
    Wraps another pretrained config with attributes we need
    """
    def __init__(self,
                 config: PretrainedConfig,
                 n_relations: int,
                 att_type: AttentionType = 'edge_as_att'):
        self.__wrapped = config
        self.n_relations = n_relations
        self.att_type = att_type

    def __getattr__(self, name):
        return getattr(self.__wrapped, name)
