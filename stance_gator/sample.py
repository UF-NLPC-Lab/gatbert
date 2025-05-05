import dataclasses
from typing import List, Optional
# Local
from .constants import Stance, EzstanceDomains

@dataclasses.dataclass
class Sample:
    context: str
    target: str
    stance: Stance
    domain: Optional[EzstanceDomains] = None

@dataclasses.dataclass
class PretokenizedSample:
    context: List[str]
    target: List[str]
    stance: Stance
    domain: Optional[EzstanceDomains] = None