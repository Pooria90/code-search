import re

from tree_sitter import Node

_DEF_TYPES = {"function_definition", "class_definition"}
_INDENT = "    "


def unwrap(node: Node) -> Node:
    """Return the function_definition/class_definition a decorated_definition
    wraps, or the node itself if it isn't decorated."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        assert inner is not None
        return inner
    return node


def collect_definitions(node: Node) -> list[Node]:
    """Collect def/class nodes under `node`, descending into class bodies so
    methods are collected too. Each result is the outermost node for that
    definition (a decorated_definition if decorated, else the bare def)."""
    result: list[Node] = []
    for child in node.children:
        if child.type == "decorated_definition" or child.type in _DEF_TYPES:
            result.append(child)
            definition = unwrap(child)
            if definition.type == "class_definition":
                body = definition.child_by_field_name("body")
                if body is not None:
                    result.extend(collect_definitions(body))
        else:
            result.extend(collect_definitions(child))
    return result


def enclosing_class(node: Node) -> Node | None:
    """Nearest enclosing class's outer node (decorated_definition if the
    class is decorated, else class_definition), or None if `node` isn't a
    direct member of a class body (a function body in between doesn't
    count — nested defs inside a method aren't methods themselves)."""
    current = node.parent
    while current is not None:
        candidate = unwrap(current)
        if candidate.type == "class_definition":
            outer = current.parent
            if outer is not None and outer.type == "decorated_definition":
                return outer
            return current
        if candidate.type == "function_definition":
            return None
        current = current.parent
    return None


def get_decorators(node: Node) -> list[str]:
    if node.type != "decorated_definition":
        return []
    return [child.text.decode("utf-8", errors="replace") for child in node.children if child.type == "decorator"]


def get_signature(definition: Node, source: bytes) -> str:
    """Text from the start of the def/class up to (not including) its body."""
    body = definition.child_by_field_name("body")
    end = body.start_byte if body is not None else definition.end_byte
    return source[definition.start_byte : end].decode("utf-8", errors="replace").rstrip()


def qualified_name(node: Node) -> str:
    """Unwrap only the starting node, then walk raw parents. Unwrapping every
    step would count a decorated definition twice, since the wrapper and the
    definition it wraps both resolve to the same name."""
    parts = []
    current: Node | None = unwrap(node)
    while current is not None:
        if current.type in _DEF_TYPES:
            name_node = current.child_by_field_name("name")
            if name_node is not None:
                parts.append(name_node.text.decode("utf-8", errors="replace"))
        current = current.parent
    return ".".join(reversed(parts))


def _leading_docstring(children: list[Node]) -> str | None:
    if not children:
        return None
    first = children[0]
    if first.type != "expression_statement" or not first.children:
        return None
    string_node = first.children[0]
    if string_node.type != "string":
        return None
    content_node = next((c for c in string_node.children if c.type == "string_content"), None)
    if content_node is None:
        return None
    return content_node.text.decode("utf-8", errors="replace").strip()


def get_docstring(definition: Node) -> str | None:
    body = definition.child_by_field_name("body")
    if body is None:
        return None
    return _leading_docstring(list(body.children))


def get_module_docstring(root: Node) -> str | None:
    return _leading_docstring(list(root.children))


def _collapse_signature(signature: str) -> str:
    """Flatten a multi-line def/class signature onto one line. Signatures keep
    their original interior indentation, which reads misaligned once re-indented
    into a skeleton."""
    collapsed = re.sub(r"\s+", " ", signature).strip()
    collapsed = re.sub(r"\(\s+", "(", collapsed)
    collapsed = re.sub(r"\s+([),:])", r"\1", collapsed)
    return collapsed


_SUMMARY_MAX_CHARS = 200
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _docstring_summary(definition: Node) -> str | None:
    """Docstring flattened to one line, cut at the first sentence end within a
    character budget. Deliberately a rough heuristic — see
    resources/design/python_chunking.md for what needs tuning here."""
    docstring = get_docstring(definition)
    if docstring is None:
        return None

    flat = re.sub(r"\s+", " ", docstring).strip()
    if not flat:
        return None

    window = flat[:_SUMMARY_MAX_CHARS]
    sentence_end = _SENTENCE_END.search(window)
    if sentence_end is not None:
        return window[: sentence_end.end()]
    return window.rstrip() + "..." if len(flat) > _SUMMARY_MAX_CHARS else window


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _is_skeleton_attribute(member: Node, source: bytes) -> bool:
    """Class-level assignment worth keeping: anything but dunders and privates.
    Annotation presence tracks the author's typing style, not API relevance —
    see resources/design/python_chunking.md."""
    if member.type != "expression_statement" or not member.children:
        return False
    assignment = member.children[0]
    if assignment.type != "assignment":
        return False
    left = assignment.child_by_field_name("left")
    if left is None or left.type != "identifier":
        return False
    return not _node_text(left, source).startswith("_")


def class_skeleton(node: Node, source: bytes) -> str:
    """Class rendered as a stub: signature, docstring, public attributes, and
    member signatures with bodies elided. Keeps the coarse CLASS chunk bounded
    by API surface instead of implementation size, and stops it duplicating the
    METHOD chunks."""
    definition = unwrap(node)
    lines = [*get_decorators(node), get_signature(definition, source)]

    docstring = get_docstring(definition)
    if docstring is not None:
        lines.append(f'{_INDENT}"""{docstring}"""')

    body = definition.child_by_field_name("body")
    for member in body.children if body is not None else []:
        if _is_skeleton_attribute(member, source):
            lines.append(_INDENT + _collapse_signature(_node_text(member, source)))
            continue

        inner = unwrap(member) if member.type == "decorated_definition" else member
        if inner.type not in _DEF_TYPES:
            continue

        lines.extend(f"{_INDENT}{d}" for d in get_decorators(member))
        signature = _collapse_signature(get_signature(inner, source))
        summary = _docstring_summary(inner)
        if summary is None:
            lines.append(f"{_INDENT}{signature} ...")
        else:
            lines.append(f"{_INDENT}{signature}")
            lines.append(f'{_INDENT * 2}"""{summary}"""')

    return "\n".join(lines)
