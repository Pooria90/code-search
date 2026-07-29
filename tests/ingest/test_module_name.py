from code_search.ingest.models import ChunkKind
from code_search.ingest.python_chunker import parse_file, resolve_module_name


def _write(root, rel_path, content="x = 1\n"):
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def test_src_layout_drops_directories_above_the_import_root(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\n")
    _write(tmp_path, "src/mypkg/__init__.py")
    target = _write(tmp_path, "src/mypkg/core.py")
    assert resolve_module_name(target) == "mypkg.core"


def test_nested_packages(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\n")
    _write(tmp_path, "src/mypkg/__init__.py")
    _write(tmp_path, "src/mypkg/sub/__init__.py")
    target = _write(tmp_path, "src/mypkg/sub/thing.py")
    assert resolve_module_name(target) == "mypkg.sub.thing"


def test_namespace_package_without_init(tmp_path):
    """PEP 420 dirs are importable, so they belong in the module path — flask's
    `flask/sansio/` has no __init__.py but is imported as `flask.sansio`."""
    _write(tmp_path, "pyproject.toml", "[project]\n")
    _write(tmp_path, "src/mypkg/__init__.py")
    target = _write(tmp_path, "src/mypkg/plain/thing.py")
    assert resolve_module_name(target) == "mypkg.plain.thing"


def test_package_init_resolves_to_the_package(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\n")
    target = _write(tmp_path, "src/mypkg/__init__.py")
    assert resolve_module_name(target) == "mypkg"


def test_walk_stops_at_project_marker(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\n")
    target = _write(tmp_path, "script.py")
    assert resolve_module_name(target) == "script"


def test_module_chunk_uses_resolved_name(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\n")
    _write(tmp_path, "src/mypkg/__init__.py")
    target = _write(tmp_path, "src/mypkg/core.py", "import os\n\nCONST = 1\n")

    chunks = parse_file(target, repo="r", count_tokens=lambda text: len(text.split()))
    module_chunk = next(c for c in chunks if c.kind is ChunkKind.MODULE)

    assert module_chunk.qualified_name == "mypkg.core"


def test_module_name_is_independent_of_checkout_location(tmp_path):
    """Same package under two different parent directories must resolve to the
    same module name — otherwise the name encodes where the repo was cloned."""
    names = []
    for outer in ("checkout_a", "some/deeper/checkout_b"):
        root = tmp_path / outer
        _write(root, "pyproject.toml", "[project]\n")
        _write(root, "src/mypkg/__init__.py")
        target = _write(root, "src/mypkg/core.py")
        names.append(resolve_module_name(target))

    assert names[0] == names[1] == "mypkg.core"
