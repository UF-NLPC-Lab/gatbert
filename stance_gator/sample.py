import dataclasses
from typing import List, Optional
# Local
from .constants import BaseStance

@dataclasses.dataclass
class Sample:
    context: str | List[str]
    target: str | List[str]
    stance: BaseStance
    is_split_into_words: bool
    lang: Optional[str] = None
    weight: Optional[float] = None