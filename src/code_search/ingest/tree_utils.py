from tree_sitter import Node

_DEF_TYPES = {"function_definition", "class_definition"}


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
    parts = []
    current: Node | None = node
    while current is not None:
        candidate = unwrap(current)
        if candidate.type in _DEF_TYPES:
            name_node = candidate.child_by_field_name("name")
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
