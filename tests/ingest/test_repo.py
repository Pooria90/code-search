import pytest

from code_search.ingest.repo import ingest_repo, is_test_file, iter_python_files
from pathlib import Path


def _write(root: Path, rel: str, content: str = "VALUE = 1\n") -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "myrepo"
    _write(root, "pyproject.toml", "[project]\n")
    _write(root, "src/mypkg/__init__.py", "")
    _write(root, "src/mypkg/core.py", "import os\n\n\ndef work():\n    return os.getcwd()\n")
    _write(root, "tests/test_core.py", "def test_work():\n    assert True\n")
    return root


def _counted(root, **kwargs):
    return ingest_repo(root, count_tokens=lambda text: len(text.split()), **kwargs)


def test_finds_python_files_only(repo):
    _write(repo, "README.md", "# docs\n")
    _write(repo, "src/mypkg/data.json", "{}\n")
    found = {p.name for p in iter_python_files(repo)}
    assert found == {"__init__.py", "core.py", "test_core.py"}


def test_excluded_directories_are_pruned(repo):
    _write(repo, ".venv/lib/thing.py")
    _write(repo, "src/mypkg/__pycache__/core.py")
    _write(repo, "build/generated.py")
    _write(repo, "mypkg.egg-info/meta.py")
    _write(repo, ".git/hooks/thing.py")

    found = {p.name for p in iter_python_files(repo)}
    assert found == {"__init__.py", "core.py", "test_core.py"}


def test_traversal_order_is_deterministic(repo):
    for name in ("z_last.py", "a_first.py", "m_middle.py"):
        _write(repo, f"src/mypkg/{name}")

    first = [str(p) for p in iter_python_files(repo)]
    assert first == [str(p) for p in iter_python_files(repo)]

    # depth-first, so the guarantee is ordering within a directory, not globally
    in_pkg = [Path(p).name for p in first if Path(p).parent.name == "mypkg"]
    assert in_pkg == sorted(in_pkg)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("tests/test_core.py", True),
        ("src/mypkg/core_test.py", True),
        ("examples/tutorial/tests/conftest.py", True),
        ("test/helpers.py", True),
        ("src/mypkg/core.py", False),
        ("src/mypkg/contest.py", False),
        ("src/latest/thing.py", False),
    ],
)
def test_is_test_file(path, expected):
    assert is_test_file(Path(path)) is expected


def test_include_tests_false_skips_them(repo):
    assert {p.name for p in iter_python_files(repo, include_tests=False)} == {"__init__.py", "core.py"}


def test_ingest_produces_chunks_with_repo_relative_paths(repo):
    result = _counted(repo)
    assert result.file_count == 3
    assert result.chunks
    assert all(c.repo == "myrepo" for c in result.chunks)
    assert "src/mypkg/core.py" in {c.path for c in result.chunks}
    assert not any(Path(c.path).is_absolute() for c in result.chunks)


def test_oversized_files_are_skipped_not_failed(repo):
    _write(repo, "src/mypkg/huge.py", "VALUE = 1\n" * 5000)
    result = _counted(repo, max_file_bytes=1000)

    assert [Path(s.path).name for s in result.skipped] == ["huge.py"]
    assert result.failures == []
    assert "huge.py" not in {Path(c.path).name for c in result.chunks}


def test_unreadable_file_is_recorded_and_does_not_abort(repo, monkeypatch):
    real_parse = ingest_repo.__globals__["parse_file"]

    def flaky(path, **kwargs):
        if Path(path).name == "core.py":
            raise OSError("boom")
        return real_parse(path, **kwargs)

    monkeypatch.setitem(ingest_repo.__globals__, "parse_file", flaky)
    result = _counted(repo)

    assert [Path(f.path).name for f in result.failures] == ["core.py"]
    assert "boom" in result.failures[0].reason
    # the other files still made it through
    assert result.file_count == 2
    assert result.chunks


def test_syntax_errors_are_surfaced(repo):
    _write(repo, "src/mypkg/broken.py", "def oops(:\n    pass\n")
    result = _counted(repo)
    assert result.files_with_syntax_errors == {"src/mypkg/broken.py"}


def test_empty_repo_yields_empty_result(tmp_path):
    (tmp_path / "empty").mkdir()
    result = _counted(tmp_path / "empty")
    assert result.chunks == [] and result.file_count == 0
