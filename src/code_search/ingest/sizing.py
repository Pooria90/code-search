from tree_sitter import Node


def split_if_needed(node: Node) -> list[Node]:
    """NOTE: Placeholder hook point. No token-size-aware splitting yet — every node
    passes through unchanged.
    Real logic (recurse into a node's own `body` children once it
    exceeds max_tokens) lands once an embedding model / tokenizer is chosen"""
    return [node]
