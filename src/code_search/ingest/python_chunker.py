import hashlib
from pathlib import Path

import tree_sitter_python as tspython
from loguru import logger
from tree_sitter import Language, Node, Parser

from code_search.ingest.models import Chunk, ChunkKind
from code_search.ingest.sizing import split_if_needed
from code_search.ingest.tree_utils import (
    collect_definitions,
    enclosing_class,
    get_decorators,
    get_docstring,
    get_module_docstring,
    get_signature,
    qualified_name,
    unwrap,
)

_LANGUAGE = Language(tspython.language())
_DEF_TYPES = {"function_definition", "class_definition"}


def _make_id(repo: str, path: str, content: str) -> str:
    digest = hashlib.sha256(f"{repo}:{path}:{content}".encode("utf-8")).hexdigest()
    return digest[:16]


def _content_for(node: Node, class_header: str | None) -> str:
    text = node.text.decode("utf-8", errors="replace")
    if class_header is None:
        return text
    return f"{class_header}\n    ...\n{text}"


def _chunk_for_definition(node: Node, source: bytes, repo: str, path: str, has_error: bool) -> Chunk:
    definition = unwrap(node)
    parent_class = enclosing_class(node)

    class_header = get_signature(unwrap(parent_class), source) if parent_class is not None else None
    kind = (
        ChunkKind.CLASS
        if definition.type == "class_definition"
        else ChunkKind.METHOD
        if parent_class is not None
        else ChunkKind.FUNCTION
    )
    content = _content_for(node, class_header)
    parent_id = _make_id(repo, path, _content_for(parent_class, None)) if parent_class is not None else None

    return Chunk(
        id=_make_id(repo, path, content),
        repo=repo,
        path=path,
        kind=kind,
        qualified_name=qualified_name(node),
        signature=get_signature(definition, source),
        docstring=get_docstring(definition),
        decorators=get_decorators(node),
        content=content,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        parent_id=parent_id,
        has_error=has_error,
    )


def _module_chunk(root: Node, source: bytes, repo: str, path: str, has_error: bool) -> Chunk | None:
    header_children = [
        child for child in root.children if child.type not in _DEF_TYPES and child.type != "decorated_definition"
    ]
    if not header_children:
        return None
    content = "\n".join(child.text.decode("utf-8", errors="replace") for child in header_children).strip()
    if not content:
        return None
    return Chunk(
        id=_make_id(repo, path, content),
        repo=repo,
        path=path,
        kind=ChunkKind.MODULE,
        qualified_name=path.replace("/", ".").removesuffix(".py"),
        signature=None,
        docstring=get_module_docstring(root),
        decorators=[],
        content=content,
        start_line=header_children[0].start_point[0] + 1,
        end_line=header_children[-1].end_point[0] + 1,
        parent_id=None,
        has_error=has_error,
    )


def parse_source(source: bytes, repo: str, path: str) -> list[Chunk]:
    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)
    root = tree.root_node
    has_error = root.has_error
    if has_error:
        logger.warning(f"{path}: parsed with syntax errors, chunks may be incomplete")

    chunks: list[Chunk] = []

    module_chunk = _module_chunk(root, source, repo, path, has_error)
    if module_chunk is not None:
        chunks.append(module_chunk)

    for node in collect_definitions(root):
        for piece in split_if_needed(node):
            chunks.append(_chunk_for_definition(piece, source, repo, path, has_error))

    return chunks


def parse_file(path: str | Path, repo: str = "") -> list[Chunk]:
    file_path = Path(path)
    source = file_path.read_bytes()
    return parse_source(source, repo=repo, path=str(file_path))
