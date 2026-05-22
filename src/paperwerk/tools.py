import inspect
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin, get_type_hints


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: dict

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai(self) -> list[dict]:
        return [t.to_openai() for t in self._tools.values()]

    def dispatch(self, ctx: Any, name: str, arguments: dict) -> Any:
        return self.get(name).func(ctx, **arguments)


def _annotation_to_schema(annotation: Any) -> dict:
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_type = args[0] if args else str
        return {"type": "array", "items": _annotation_to_schema(item_type)}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list:
        return {"type": "array"}
    return {"type": "string"}


REGISTRY = ToolRegistry()


_SECTION_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z ]*):\s*$")
_PARAM_ENTRY_RE = re.compile(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$")
_ARGS_SECTIONS = ("Args", "Arguments", "Parameters")


def _parse_google_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Return (description, param_descriptions) parsed from a Google-style docstring."""
    if not docstring:
        return "", {}
    lines = inspect.cleandoc(docstring).splitlines()

    sections: dict[str, list[str]] = {"_description": []}
    current = "_description"
    for line in lines:
        match = _SECTION_HEADER_RE.match(line)
        if match:
            current = match.group(1).strip()
            sections[current] = []
        else:
            sections[current].append(line)

    description = "\n".join(sections["_description"]).strip()

    args_lines: list[str] = []
    for key in _ARGS_SECTIONS:
        if key in sections:
            args_lines = sections[key]
            break
    if not args_lines:
        return description, {}

    args_lines = textwrap.dedent("\n".join(args_lines)).splitlines()
    params: dict[str, str] = {}
    j = 0
    while j < len(args_lines):
        line = args_lines[j]
        if line and not line[0].isspace():
            match = _PARAM_ENTRY_RE.match(line)
            if match:
                name = match.group(1)
                desc_parts = [match.group(2).strip()] if match.group(2).strip() else []
                j += 1
                while j < len(args_lines) and (
                    not args_lines[j].strip() or args_lines[j][0].isspace()
                ):
                    if args_lines[j].strip():
                        desc_parts.append(args_lines[j].strip())
                    j += 1
                params[name] = " ".join(desc_parts)
                continue
        j += 1

    return description, params


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    registry: ToolRegistry = REGISTRY,
) -> Callable:
    """Register a function as a tool.

    The function's Google-style docstring supplies the tool description and
    per-parameter descriptions. The first parameter (the context) is excluded
    from the generated JSON schema. Usable as `@tool` or `@tool(name=...)`.
    """

    def decorator(fn: Callable) -> Callable:
        description, params_doc = _parse_google_docstring(fn.__doc__)
        sig = inspect.signature(fn)
        type_hints = get_type_hints(fn)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for pname, param in list(sig.parameters.items())[1:]:
            schema = _annotation_to_schema(type_hints.get(pname, str))
            if pname in params_doc:
                schema["description"] = params_doc[pname]
            properties[pname] = schema
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        registry.register(
            Tool(
                name=name or fn.__name__,
                description=description,
                func=fn,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
        )
        return fn

    if func is not None and callable(func):
        return decorator(func)
    return decorator
