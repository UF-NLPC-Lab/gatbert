import dataclasses
from typing import List, Any, Optional
# Local
from .constants import BaseStance

@dataclasses.dataclass
class Sample:
    context: str | List[str]
    target: str | List[str]
    stance: BaseStance
    is_split_into_words: bool
    domain: Optional[Any] = None
