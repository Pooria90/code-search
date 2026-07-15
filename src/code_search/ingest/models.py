from dataclasses import dataclass
from enum import Enum


class ChunkKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


@dataclass
class Chunk:
    id: str
    repo: str
    path: str
    kind: ChunkKind
    qualified_name: str
    signature: str | None
    docstring: str | None
    decorators: list[str]
    content: str
    start_line: int
    end_line: int
    parent_id: str | None
