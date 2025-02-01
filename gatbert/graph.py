# STL
from typing import List, Tuple, Dict, Any
import dataclasses

EdgeList = List[Tuple[int, int, int]]
NodeList = List[int]

@dataclasses.dataclass
class CNGraph:
    tok2id: Dict[str, int]
    id2uri: Dict[int, str]
    adj: Dict[int, List[Tuple[int, int]]]

    @staticmethod
    def from_json(json_data: Dict[str, Any]):
        return CNGraph(
            tok2id=json_data['tok2id'],
            id2uri={int(k):v for k,v in json_data['id2uri'].items()},
            adj   ={int(k):v for k,v in json_data['adj'].items()}
        )
