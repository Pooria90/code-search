import hashlib
from collections.abc import Callable
from pathlib import Path

import tree_sitter_python as tspython
from loguru import logger
from tree_sitter import Language, Node, Parser

from code_search.ingest.models import Chunk, ChunkKind
from code_search.ingest.sizing import MAX_TOKENS, split_if_needed
from code_search.ingest.tree_utils import (
    class_skeleton,
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


_SOURCE_ROOT_NAMES = {"src", "lib"}
_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")
_REPO_MARKERS = (".git", *_PROJECT_MARKERS)


def find_repo_root(path: str | Path) -> Path | None:
    """Nearest ancestor holding a repo marker. Innermost wins, so a vendored
    checkout resolves to its own root rather than the outer project's."""
    for candidate in Path(path).resolve().parents:
        if any((candidate / marker).exists() for marker in _REPO_MARKERS):
            return candidate
    return None


def resolve_module_name(path: str | Path) -> str:
    """Dotted module name for a file, walking up the package tree so names are
    free of wherever the repo happens to sit on disk: `<anything>/src/flask/
    app.py` is `flask.app`.

    A parent counts as a package if it has an `__init__.py`, or — for PEP 420
    namespace packages like `flask/sansio/` — if it merely looks like one. The
    walk stops at a source root or a directory holding a project marker, since
    those sit above the import root."""
    file_path = Path(path)
    parts = [file_path.stem]

    parent = file_path.parent
    while parent != parent.parent:
        if not (parent / "__init__.py").is_file():
            if (
                parent.name in _SOURCE_ROOT_NAMES
                or not parent.name.isidentifier()
                or any((parent / marker).is_file() for marker in _PROJECT_MARKERS)
            ):
                break
        parts.append(parent.name)
        parent = parent.parent

    if parts[0] == "__init__":
        parts.pop(0)
    return ".".join(reversed(parts))


def _fallback_module_name(path: str) -> str:
    """Used when there's no file to inspect (in-memory `parse_source`)."""
    name = path.replace("\\", "/").removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def _embed_content(node: Node, source: bytes, class_header: str | None = None) -> str:
    """The text a definition contributes to the index. Classes become skeletons
    so the coarse chunk doesn't duplicate its own METHOD chunks; methods get
    their class header prepended for scope.

    Both a class chunk's own id and its methods' parent_id are hashed from this,
    so the two agree by construction rather than by coincidence."""
    if unwrap(node).type == "class_definition":
        return class_skeleton(node, source)
    text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
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
    content = _embed_content(node, source, class_header)
    parent_id = _make_id(repo, path, _embed_content(parent_class, source)) if parent_class is not None else None

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


def _module_chunk(
    root: Node, source: bytes, repo: str, path: str, module_name: str, has_error: bool
) -> Chunk | None:
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
        qualified_name=module_name,
        signature=None,
        docstring=get_module_docstring(root),
        decorators=[],
        content=content,
        start_line=header_children[0].start_point[0] + 1,
        end_line=header_children[-1].end_point[0] + 1,
        parent_id=None,
        has_error=has_error,
    )


def parse_source(
    source: bytes,
    repo: str,
    path: str,
    max_tokens: int = MAX_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
    module_name: str | None = None,
) -> list[Chunk]:
    if count_tokens is None:
        from code_search.ingest.tokenizer import count_tokens as count_tokens_default

        count_tokens = count_tokens_default
    if module_name is None:
        module_name = _fallback_module_name(path)

    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)
    root = tree.root_node
    has_error = root.has_error
    if has_error:
        logger.warning(f"{path}: parsed with syntax errors, chunks may be incomplete")

    chunks: list[Chunk] = []

    module_chunk = _module_chunk(root, source, repo, path, module_name, has_error)
    if module_chunk is not None:
        chunks.append(module_chunk)

    for node in collect_definitions(root):
        for piece in split_if_needed(node, max_tokens, count_tokens):
            chunks.append(_chunk_for_definition(piece, source, repo, path, has_error))

    return chunks


def parse_file(
    path: str | Path,
    repo: str = "",
    max_tokens: int = MAX_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
    repo_root: str | Path | None = None,
) -> list[Chunk]:
    """Chunk a file. `repo` and `path` are recorded relative to the repository so
    chunk ids don't depend on where it sits on disk; pass `repo_root` to pin it,
    otherwise the nearest repo marker above the file is used."""
    file_path = Path(path)
    source = file_path.read_bytes()

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root(file_path)
    relative_path = file_path
    if root is not None:
        try:
            relative_path = file_path.resolve().relative_to(root)
        except ValueError:
            relative_path = file_path

    return parse_source(
        source,
        repo=repo or (root.name if root is not None else ""),
        # posix + repo-relative so the same file hashes identically across
        # machines, checkout locations, and platforms
        path=relative_path.as_posix(),
        max_tokens=max_tokens,
        count_tokens=count_tokens,
        # resolved from the real location on disk, not the relative path
        module_name=resolve_module_name(file_path),
    )
