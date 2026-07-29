from functools import lru_cache

from tokenizers import Tokenizer

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)
