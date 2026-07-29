from code_search.ingest.python_chunker import find_repo_root, parse_file


def _make_repo(root, marker="pyproject.toml"):
    (root / marker).parent.mkdir(parents=True, exist_ok=True)
    (root / marker).write_text("[project]\n")
    target = root / "src" / "mypkg" / "core.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "__init__.py").write_text("")
    target.write_text("import os\n\n\ndef work():\n    return os.getcwd()\n")
    return target


def _chunks(target, **kwargs):
    return parse_file(target, count_tokens=lambda text: len(text.split()), **kwargs)


def test_path_is_recorded_relative_to_the_repo(tmp_path):
    target = _make_repo(tmp_path / "checkout")
    assert all(c.path == "src/mypkg/core.py" for c in _chunks(target))


def test_repo_defaults_to_the_repo_directory_name(tmp_path):
    target = _make_repo(tmp_path / "flask")
    assert all(c.repo == "flask" for c in _chunks(target))


def test_explicit_repo_wins_over_the_directory_name(tmp_path):
    target = _make_repo(tmp_path / "flask")
    assert all(c.repo == "pallets/flask" for c in _chunks(target, repo="pallets/flask"))


def test_ids_are_independent_of_checkout_location(tmp_path):
    """The point of hashing content: the same file in the same repo must get the
    same id wherever the repo is cloned, or incremental re-indexing re-embeds
    everything."""
    ids = []
    for outer in ("checkout_a", "some/deeper/place/checkout_b"):
        target = _make_repo(tmp_path / outer / "myrepo")
        ids.append([c.id for c in _chunks(target)])

    assert ids[0] == ids[1]


def test_explicit_repo_root_overrides_detection(tmp_path):
    target = _make_repo(tmp_path / "checkout")
    chunks = _chunks(target, repo_root=tmp_path / "checkout" / "src")
    assert all(c.path == "mypkg/core.py" for c in chunks)


def test_innermost_repo_marker_wins(tmp_path):
    """A vendored checkout inside another project resolves to its own root."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    target = _make_repo(tmp_path / "vendor" / "inner")

    assert find_repo_root(target) == (tmp_path / "vendor" / "inner").resolve()
    assert all(c.path == "src/mypkg/core.py" for c in _chunks(target))


def test_git_directory_counts_as_a_repo_root(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    target = root / "pkg" / "mod.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")

    assert find_repo_root(target) == root.resolve()
    assert all(c.path == "pkg/mod.py" for c in _chunks(target))


def test_no_marker_found_leaves_the_path_alone(tmp_path, monkeypatch):
    target = tmp_path / "loose.py"
    target.write_text("VALUE = 1\n")
    monkeypatch.setattr("code_search.ingest.python_chunker.find_repo_root", lambda _: None)

    chunks = _chunks(target)
    assert all(c.path == target.as_posix() for c in chunks)
    assert all(c.repo == "" for c in chunks)
