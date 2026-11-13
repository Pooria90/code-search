import numpy as np
import pytest

from code_search.embed import Embedder, chunk_to_text
from code_search.ingest.models import Chunk, ChunkKind


class FakeModel:
    """Stands in for SentenceTransformer so these tests need no model download."""

    max_seq_length = 8192

    def __init__(self, dimension: int | None = 4):
        self.dimension = dimension
        self.calls: list[dict] = []

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        # float64 on purpose: the embedder should cast down to float32
        return np.array([[float(len(t)), 1.0, 2.0, 3.0] for t in texts], dtype=np.float64)


def _embedder(dimension: int | None = 4, **kwargs) -> tuple[Embedder, FakeModel]:
    embedder = Embedder(**kwargs)
    fake = FakeModel(dimension)
    embedder.__dict__["model"] = fake  # cached_property override
    return embedder, fake


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        id="abc123",
        repo="flask",
        path="src/flask/app.py",
        kind=ChunkKind.CLASS,
        qualified_name="Flask",
        signature="class Flask(App):",
        docstring="The flask object.",
        decorators=[],
        content="class Flask(App):\n    ...",
        start_line=1,
        end_line=2,
        parent_id=None,
    )
    return Chunk(**{**defaults, **overrides})


def test_header_carries_the_metadata_a_query_might_match():
    text = chunk_to_text(_chunk())
    header = text.splitlines()[0]
    assert header.startswith("#")
    for expected in ("flask", "src/flask/app.py", "class", "Flask"):
        assert expected in header
    assert text.endswith("class Flask(App):\n    ...")


def test_header_can_be_disabled():
    chunk = _chunk()
    assert chunk_to_text(chunk, include_header=False) == chunk.content


def test_encode_returns_float32_matrix():
    embedder, _ = _embedder()
    vectors = embedder.encode(["a", "bb", "ccc"])
    assert vectors.shape == (3, 4)
    assert vectors.dtype == np.float32


def test_encode_empty_input_keeps_the_dimension():
    embedder, fake = _embedder()
    vectors = embedder.encode([])
    assert vectors.shape == (0, 4)
    assert vectors.dtype == np.float32
    assert fake.calls == []


def test_encode_requests_normalized_vectors_by_default():
    embedder, fake = _embedder()
    embedder.encode(["a"])
    assert fake.calls[0]["normalize_embeddings"] is True


def test_normalization_can_be_disabled():
    embedder, fake = _embedder(normalize=False)
    embedder.encode(["a"])
    assert fake.calls[0]["normalize_embeddings"] is False


def test_encode_chunks_uses_header_setting():
    embedder, fake = _embedder(include_header=False)
    chunk = _chunk()
    embedder.encode_chunks([chunk])
    assert fake.calls[0]["texts"] == [chunk.content]


def test_encode_query_returns_a_single_vector():
    embedder, _ = _embedder()
    vector = embedder.encode_query("how do I register a blueprint")
    assert vector.shape == (4,)


def test_batch_size_is_passed_through():
    embedder, fake = _embedder()
    embedder.encode(["a", "b"], batch_size=2)
    assert fake.calls[0]["batch_size"] == 2


def test_missing_dimension_is_an_explicit_error():
    embedder, _ = _embedder(dimension=None)
    with pytest.raises(RuntimeError, match="embedding dimension"):
        _ = embedder.dimension
