import dataclasses
from typing import List
# Local
from .constants import Stance

@dataclasses.dataclass
class Sample:
    context: str
    target: str
    stance: Stance

@dataclasses.dataclass
class PretokenizedSample:
    context: List[str]
    target: List[str]
    stance: Stance

