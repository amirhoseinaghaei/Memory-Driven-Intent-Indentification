from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EdgeSpec:
    from_id: str
    to_id: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)