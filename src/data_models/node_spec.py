from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class NodeSpec:
    id: str
    layer: int
    label: str
    data: str
    embedding: Optional[List[float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)


