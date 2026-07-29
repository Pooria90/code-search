from collections.abc import Callable

from tree_sitter import Node

MAX_TOKENS = 1200
MIN_TOKENS = 50


def split_if_needed(node: Node, max_tokens: int, count_tokens: Callable[[str], int]) -> list[Node]:
    """NOTE: Placeholder hook point. No token-size-aware splitting yet — every node
    passes through unchanged.
    Real logic (recurse into a node's own `body` children once it
    exceeds max_tokens) still to come; MAX_TOKENS/MIN_TOKENS need an eval set to tune."""
    return [node]
