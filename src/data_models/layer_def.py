from dataclasses import dataclass
from typing import Set, Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class LayerDef:
    index: int
    name: str
