from typing import Optional
# 3rd Party
from transformers import PretrainedConfig
# Local
from .types import AttentionType
from .constants import DEFAULT_MODEL

class GatbertConfig:
    """
    Wraps another pretrained config with attributes we need
    """
    def __init__(self,
                 config: PretrainedConfig,
                 n_relations: int,
                 num_graph_layers: Optional[int] = None,
                 att_type: AttentionType = 'edge_as_att'):
        self.wrapped = config
        self.__num_graph_layers = num_graph_layers
        self.n_relations = n_relations
        self.att_type = att_type

    @property
    def num_graph_layers(self):
        return self.__num_graph_layers if self.__num_graph_layers is not None else self.wrapped.num_hidden_layers

    def __getattr__(self, name):
        return getattr(self.wrapped, name)
