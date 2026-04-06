from dataclasses import dataclass
from typing import Set, Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class RelationDef:
    source: str
    target: str
    rel_type: str


