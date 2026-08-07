import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from code_search.ingest.models import Chunk
from code_search.ingest.python_chunker import parse_file
from code_search.ingest.sizing import MAX_TOKENS

EXCLUDED_DIRS = {
    "__pycache__",
    "build",
    "dist",
    "env",
    "site-packages",
    "venv",
}

MAX_FILE_BYTES = 1_000_000


@dataclass
class SkippedFile:
    path: str
    reason: str


@dataclass
class IngestResult:
    chunks: list[Chunk] = field(default_factory=list)
    file_count: int = 0
    failures: list[SkippedFile] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)

    @property
    def files_with_syntax_errors(self) -> set[str]:
        return {chunk.path for chunk in self.chunks if chunk.has_error}


def _is_excluded_dir(name: str) -> bool:
    return name.startswith(".") or name in EXCLUDED_DIRS or name.endswith(".egg-info")


def is_test_file(path: Path) -> bool:
    if path.name.startswith("test_") or path.stem.endswith("_test"):
        return True
    return any(part in {"test", "tests"} for part in path.parts)


def iter_python_files(root: str | Path, include_tests: bool = True) -> Iterator[Path]:
    """Walk `root` for .py files, pruning excluded directories as we go.

    Uses os.walk rather than rglob so whole trees (.git, .venv, node_modules)
    are never descended into, and sorts at each level so the traversal order is
    reproducible."""
    root_path = Path(root)
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = sorted(d for d in dirnames if not _is_excluded_dir(d))
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            file_path = Path(dirpath) / filename
            if not include_tests and is_test_file(file_path.relative_to(root_path)):
                continue
            yield file_path


def ingest_repo(
    root: str | Path,
    repo: str = "",
    include_tests: bool = True,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_tokens: int = MAX_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
) -> IngestResult:
    """Chunk every Python file in a repository.

    One bad file doesn't abort the run — read and parse errors are collected in
    the result so the caller can see what was missed."""
    root_path = Path(root).resolve()
    result = IngestResult()

    for file_path in iter_python_files(root_path, include_tests=include_tests):
        size = file_path.stat().st_size
        if size > max_file_bytes:
            result.skipped.append(
                SkippedFile(str(file_path), f"{size / 1_000_000:.1f} MB exceeds max_file_bytes")
            )
            continue

        try:
            result.chunks.extend(
                parse_file(
                    file_path,
                    repo=repo,
                    max_tokens=max_tokens,
                    count_tokens=count_tokens,
                    repo_root=root_path,
                )
            )
        except OSError as exc:
            result.failures.append(SkippedFile(str(file_path), str(exc)))
            continue

        result.file_count += 1

    logger.info(
        f"{root_path.name}: {len(result.chunks)} chunks from {result.file_count} files "
        f"({len(result.skipped)} skipped, {len(result.failures)} failed)"
    )
    return result
