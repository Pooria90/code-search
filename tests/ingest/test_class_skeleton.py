from code_search.ingest.models import ChunkKind
from code_search.ingest.python_chunker import parse_source

SOURCE = '''
@dataclass
class Widget(Base):
    """A widget.

    Longer explanation that should stay in the class docstring.
    """

    name: str = "w"
    retries = 3
    _internal = "hidden"
    __slots__ = ("name",)

    @property
    def label(self) -> str:
        """The display label."""
        secret = compute_something(self.name)
        return secret.upper()

    async def refresh(self,
                      force: bool = False) -> None:
        """
        Reload the widget from disk.
        """
        do_work()

    def bare(self):
        helper()

    class Meta:
        ordering = "name"
'''


def _skeleton():
    chunks = parse_source(
        SOURCE.encode("utf-8"),
        repo="test-repo",
        path="sample.py",
        count_tokens=lambda text: len(text.split()),
    )
    return next(c for c in chunks if c.qualified_name == "Widget")


def test_method_bodies_are_excluded():
    content = _skeleton().content
    assert "compute_something" not in content
    assert "secret.upper()" not in content
    assert "do_work()" not in content
    assert "helper()" not in content


def test_member_signatures_are_kept():
    content = _skeleton().content
    assert "def label(self) -> str:" in content
    assert "def bare(self): ..." in content


def test_class_decorators_and_signature_are_kept():
    content = _skeleton().content
    assert content.startswith("@dataclass\nclass Widget(Base):")


def test_member_decorators_are_kept():
    assert "@property" in _skeleton().content


def test_full_class_docstring_is_kept():
    content = _skeleton().content
    assert "A widget." in content
    assert "Longer explanation that should stay in the class docstring." in content


def test_member_docstring_is_summarized_to_one_sentence():
    content = _skeleton().content
    assert '"""The display label."""' in content
    # wrapped docstring flattened onto one line, not left as a fragment
    assert '"""Reload the widget from disk."""' in content


def test_public_attributes_kept_regardless_of_annotation():
    content = _skeleton().content
    assert 'name: str = "w"' in content
    assert "retries = 3" in content


def test_dunder_and_private_attributes_are_dropped():
    content = _skeleton().content
    assert "__slots__" not in content
    assert "_internal" not in content


def test_multiline_signature_is_collapsed():
    content = _skeleton().content
    assert "async def refresh(self, force: bool = False) -> None:" in content


def test_nested_class_appears_as_member_and_gets_its_own_chunk():
    chunks = parse_source(
        SOURCE.encode("utf-8"),
        repo="test-repo",
        path="sample.py",
        count_tokens=lambda text: len(text.split()),
    )
    skeleton = next(c for c in chunks if c.qualified_name == "Widget").content
    assert "class Meta: ..." in skeleton
    assert any(c.qualified_name == "Widget.Meta" and c.kind == ChunkKind.CLASS for c in chunks)


def test_methods_still_link_to_the_class_chunk():
    """Regression guard: class content is now a skeleton, so a method's
    parent_id must be hashed from the skeleton too, not the full class body."""
    chunks = parse_source(
        SOURCE.encode("utf-8"),
        repo="test-repo",
        path="sample.py",
        count_tokens=lambda text: len(text.split()),
    )
    widget = next(c for c in chunks if c.qualified_name == "Widget")
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD and c.qualified_name.startswith("Widget.")]

    assert methods
    ids = {c.id for c in chunks}
    for method in methods:
        assert method.parent_id == widget.id
        assert method.parent_id in ids


def test_skeleton_is_smaller_than_the_class_body():
    widget = _skeleton()
    assert len(widget.content) < len(SOURCE)
