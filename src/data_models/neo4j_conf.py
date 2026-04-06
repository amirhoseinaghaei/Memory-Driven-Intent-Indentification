from dataclasses import dataclass
from typing import Set, Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
