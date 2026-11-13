from collections.abc import Sequence
from functools import cached_property

import numpy as np
from loguru import logger

from code_search.ingest.models import Chunk
from code_search.ingest.tokenizer import EMBEDDING_MODEL

# Measured on CPU over a representative sample: 4 is ~2.4x faster than 16 and
# ~1.2x faster than 8. Batches pad to their longest member, and chunk lengths
# vary by two orders of magnitude, so large batches mostly compute padding.
DEFAULT_BATCH_SIZE = 4


def chunk_to_text(chunk: Chunk, include_header: bool = True) -> str:
    """The text an embedding is actually built from.

    Natural-language queries lean on names and paths more than on syntax, so by
    default the code is prefixed with where it came from. Written as a comment
    line to stay in-distribution for a code-trained model. Whether the header
    helps is an open question — keep it switchable and settle it with the eval
    set."""
    if not include_header:
        return chunk.content
    header = f"# {chunk.repo} · {chunk.path} · {chunk.kind.value} {chunk.qualified_name}"
    return f"{header}\n{chunk.content}"


class Embedder:
    """Wraps the sentence-transformers model. The model is loaded on first use,
    so importing this module (or constructing an Embedder) stays cheap."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
        include_header: bool = True,
        normalize: bool = True,
        num_threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.include_header = include_header
        self.normalize = normalize
        # Opt-in: torch defaults to half the available cores, which costs ~40%
        # throughput here. Left off by default rather than mutating global torch
        # state behind the caller's back.
        if num_threads is not None:
            import torch

            torch.set_num_threads(num_threads)

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        logger.info(f"loading {self.model_name}")
        model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
        logger.info(f"loaded: dim={model.get_embedding_dimension()} max_seq={model.max_seq_length}")
        return model

    @cached_property
    def dimension(self) -> int:
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"{self.model_name} did not report an embedding dimension")
        return dimension

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to a (len(texts), dimension) float32 array."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_chunks(
        self,
        chunks: Sequence[Chunk],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        texts = [chunk_to_text(chunk, include_header=self.include_header) for chunk in chunks]
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query to a single (dimension,) vector.

        No query prefix: jina-embeddings-v2 is symmetric, unlike models such as
        E5 that need `query:` / `passage:` markers."""
        return self.encode([query])[0]
