from code_search.ingest.models import ChunkKind
from code_search.ingest.python_chunker import parse_source


def _chunks(source: str):
    """Chunk with a trivial token counter so tests don't need the embedding model."""
    return parse_source(
        source.encode("utf-8"),
        repo="test-repo",
        path="sample.py",
        count_tokens=lambda text: len(text.split()),
    )


def test_decorators_are_preserved_on_functions_and_methods():
    source = """
@app.route("/")
def index():
    return "hi"

class Foo:
    @staticmethod
    def bar():
        return 1
"""
    chunks = _chunks(source)
    index_chunk = next(c for c in chunks if c.qualified_name == "index")
    bar_chunk = next(c for c in chunks if c.qualified_name == "Foo.bar")

    assert index_chunk.decorators == ['@app.route("/")']
    assert '@app.route("/")' in index_chunk.content
    assert bar_chunk.decorators == ["@staticmethod"]
    assert "@staticmethod" in bar_chunk.content


def test_class_emits_class_and_method_chunks():
    source = """
class Point:
    \"\"\"A point.\"\"\"

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        return (self.x**2 + self.y**2) ** 0.5
"""
    chunks = _chunks(source)
    class_chunk = next(c for c in chunks if c.kind == ChunkKind.CLASS)
    method_chunks = [c for c in chunks if c.kind == ChunkKind.METHOD]

    assert class_chunk.qualified_name == "Point"
    assert class_chunk.docstring == "A point."
    assert {c.qualified_name for c in method_chunks} == {"Point.__init__", "Point.norm"}
    assert all(c.parent_id == class_chunk.id for c in method_chunks)
    # method content carries the enclosing class header for context
    assert all("class Point:" in c.content for c in method_chunks)


def test_decorated_class_does_not_duplicate_its_name_in_members():
    """A decorated_definition and the class_definition it wraps resolve to the
    same name, so unwrapping at every step of the parent walk yields
    'Widget.Widget.label'."""
    source = """
@dataclass
class Widget:
    def label(self):
        return 1

    class Meta:
        pass
"""
    names = {c.qualified_name for c in _chunks(source)}
    assert names == {"Widget", "Widget.label", "Widget.Meta"}


def test_top_level_function_is_not_a_method():
    source = """
def standalone():
    return 1
"""
    chunks = _chunks(source)
    func_chunk = next(c for c in chunks if c.qualified_name == "standalone")
    assert func_chunk.kind == ChunkKind.FUNCTION
    assert func_chunk.parent_id is None


def test_nested_function_inside_a_method_is_not_collected_as_a_method():
    source = """
class Foo:
    def outer(self):
        def inner():
            return 1
        return inner()
"""
    chunks = _chunks(source)
    qualified_names = {c.qualified_name for c in chunks}
    assert "Foo.outer" in qualified_names
    assert not any("inner" in name for name in qualified_names)


def test_module_chunk_captures_imports_docstring_and_constants():
    source = '''"""Module doc."""
import os
from collections import OrderedDict

DEFAULT_TIMEOUT = 30

def foo():
    pass
'''
    chunks = _chunks(source)
    module_chunk = next(c for c in chunks if c.kind == ChunkKind.MODULE)

    assert module_chunk.docstring == "Module doc."
    assert "import os" in module_chunk.content
    assert "DEFAULT_TIMEOUT = 30" in module_chunk.content
    assert "def foo" not in module_chunk.content


def test_file_with_no_module_level_statements_has_no_module_chunk():
    source = """def foo():
    pass
"""
    chunks = _chunks(source)
    assert not any(c.kind == ChunkKind.MODULE for c in chunks)


def test_malformed_source_does_not_crash_and_flags_chunks():
    source = "import os\n\ndef foo(:\n    pass\n"
    chunks = _chunks(source)
    assert isinstance(chunks, list)
    assert chunks and all(c.has_error for c in chunks)


def test_clean_source_is_not_flagged():
    source = """
import os

def foo():
    return 1
"""
    chunks = _chunks(source)
    assert chunks and not any(c.has_error for c in chunks)


def test_signature_excludes_body():
    source = """
def add(a: int, b: int = 2) -> int:
    return a + b
"""
    chunks = _chunks(source)
    func_chunk = next(c for c in chunks if c.qualified_name == "add")
    assert func_chunk.signature == "def add(a: int, b: int = 2) -> int:"


def test_chunk_ids_are_stable_across_runs():
    source = """
def foo():
    return 1
"""
    first = _chunks(source)
    second = _chunks(source)
    assert [c.id for c in first] == [c.id for c in second]
