#!/usr/bin/env python3
"""Generate deterministic Markdown documentation from the public Python API."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "generated"
MODULES = (
    ("sima_qat.qat_api", REPO_ROOT / "sima_qat" / "qat_api.py"),
    (
        "sima_qat.sima_quantizer",
        REPO_ROOT / "sima_qat" / "sima_quantizer.py",
    ),
)

Declaration = Union[
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
]
Function = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def _assignment_names(node: Union[ast.Assign, ast.AnnAssign]) -> Iterable[str]:
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    for target in targets:
        if isinstance(target, ast.Name):
            yield target.id


def _declarations(tree: ast.Module) -> Dict[str, Declaration]:
    declarations: Dict[str, Declaration] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            declarations[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _assignment_names(node):
                declarations[name] = node
    return declarations


def _literal_all(tree: ast.Module) -> Optional[Tuple[str, ...]]:
    """Return a literal ``__all__`` declaration without importing the module."""

    value_node: Optional[ast.expr] = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value_node = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value_node = node.value

    if value_node is None:
        return None

    try:
        value = ast.literal_eval(value_node)
    except (TypeError, ValueError, SyntaxError) as error:
        raise ValueError("module __all__ must be a literal list or tuple") from error

    if not isinstance(value, (list, tuple)) or not all(
        isinstance(name, str) for name in value
    ):
        raise ValueError("module __all__ must contain only string names")
    if len(value) != len(set(value)):
        raise ValueError("module __all__ must not contain duplicate names")
    return tuple(value)


def public_declarations(tree: ast.Module) -> Tuple[Tuple[str, Declaration], ...]:
    declarations = _declarations(tree)
    exports = _literal_all(tree)
    if exports is None:
        exports = tuple(
            name for name in declarations if not name.startswith("_")
        )

    missing = [name for name in exports if name not in declarations]
    if missing:
        formatted = ", ".join(repr(name) for name in missing)
        raise ValueError(
            "cannot document public names without local declarations: " + formatted
        )
    return tuple((name, declarations[name]) for name in exports)


def _expression(node: ast.AST) -> str:
    return ast.unparse(node)


def _argument(argument: ast.arg, default: Optional[ast.expr] = None) -> str:
    rendered = argument.arg
    if argument.annotation is not None:
        rendered += f": {_expression(argument.annotation)}"
    if default is not None:
        rendered += f" = {_expression(default)}"
    return rendered


def function_signature(node: Function) -> str:
    """Render a source-equivalent signature from an AST function node."""

    arguments = node.args
    positional = tuple(arguments.posonlyargs) + tuple(arguments.args)
    missing_defaults = len(positional) - len(arguments.defaults)
    defaults: Tuple[Optional[ast.expr], ...] = (
        (None,) * missing_defaults + tuple(arguments.defaults)
    )

    rendered = []
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        rendered.append(_argument(argument, default))
        if arguments.posonlyargs and index + 1 == len(arguments.posonlyargs):
            rendered.append("/")

    if arguments.vararg is not None:
        rendered.append("*" + _argument(arguments.vararg))
    elif arguments.kwonlyargs:
        rendered.append("*")

    for argument, default in zip(
        arguments.kwonlyargs, arguments.kw_defaults
    ):
        rendered.append(_argument(argument, default))

    if arguments.kwarg is not None:
        rendered.append("**" + _argument(arguments.kwarg))

    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    signature = f"{prefix}{node.name}({', '.join(rendered)})"
    if node.returns is not None:
        signature += f" -> {_expression(node.returns)}"
    return signature


def class_signature(node: ast.ClassDef) -> str:
    bases = [_expression(base) for base in node.bases]
    bases.extend(
        (
            f"{keyword.arg}={_expression(keyword.value)}"
            if keyword.arg is not None
            else f"**{_expression(keyword.value)}"
        )
        for keyword in node.keywords
    )
    if not bases:
        return node.name
    return f"{node.name}({', '.join(bases)})"


def constant_signature(name: str, node: Union[ast.Assign, ast.AnnAssign]) -> str:
    if isinstance(node, ast.AnnAssign):
        rendered = f"{name}: {_expression(node.annotation)}"
        if node.value is not None:
            rendered += f" = {_expression(node.value)}"
        return rendered
    return f"{name} = {_expression(node.value)}"


def first_sentence(doc: Optional[str]) -> str:
    if not doc:
        return ""
    line = " ".join(doc.strip().split())
    if ". " in line:
        return line.split(". ", 1)[0] + "."
    return line


def _render_declaration(name: str, node: Declaration) -> Sequence[str]:
    if isinstance(node, ast.ClassDef):
        heading = f"### Class: `{class_signature(node)}`"
        doc = ast.get_docstring(node)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        kind = (
            "Async function"
            if isinstance(node, ast.AsyncFunctionDef)
            else "Function"
        )
        heading = f"### {kind}: `{function_signature(node)}`"
        doc = ast.get_docstring(node)
    else:
        heading = f"### Constant: `{constant_signature(name, node)}`"
        doc = None

    lines = [heading, ""]
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        lines.extend([doc.strip() if doc else "No docstring available.", ""])
    return lines


def render_module_doc(module_name: str, source: Path) -> Tuple[str, str, str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    page_name = module_name.replace(".", "-") + ".md"
    lines = [
        f"# `{module_name}`",
        "",
        f"Source: `{source.relative_to(REPO_ROOT)}`",
        "",
    ]
    module_doc = ast.get_docstring(tree)
    if module_doc:
        lines.extend([module_doc.strip(), ""])

    declarations = public_declarations(tree)
    if declarations:
        lines.extend(["## Public API", ""])
    for name, node in declarations:
        lines.extend(_render_declaration(name, node))

    output = "\n".join(line.rstrip() for line in lines).rstrip() + "\n"
    return page_name, output, first_sentence(module_doc)


def render_outputs() -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    modules = []
    for module_name, source in MODULES:
        try:
            page_name, output, summary = render_module_doc(module_name, source)
        except ValueError as error:
            raise ValueError(f"{source.relative_to(REPO_ROOT)}: {error}") from error
        outputs[page_name] = output
        modules.append((module_name, page_name, summary))

    index = [
        "# QAT API Reference",
        "",
        "Generated from the QAT Python source using `scripts/generate_api_docs.py`.",
        "Do not edit these pages directly.",
        "",
        "## Modules",
        "",
    ]
    for module_name, page_name, summary in modules:
        suffix = f" - {summary}" if summary else ""
        index.append(f"- [`{module_name}`]({page_name}){suffix}")
    outputs["index.md"] = "\n".join(line.rstrip() for line in index) + "\n"
    return outputs


def write_outputs(outputs: Dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_names = set(outputs)
    for path in OUTPUT_DIR.glob("*.md"):
        if path.name not in expected_names:
            path.unlink()
    for name, content in outputs.items():
        path = OUTPUT_DIR / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")


def check_outputs(outputs: Dict[str, str]) -> bool:
    existing_names = (
        {path.name for path in OUTPUT_DIR.glob("*.md")}
        if OUTPUT_DIR.is_dir()
        else set()
    )
    expected_names = set(outputs)
    problems = []
    problems.extend(
        f"missing: docs/generated/{name}"
        for name in sorted(expected_names - existing_names)
    )
    problems.extend(
        f"unexpected: docs/generated/{name}"
        for name in sorted(existing_names - expected_names)
    )
    for name in sorted(expected_names & existing_names):
        actual = (OUTPUT_DIR / name).read_text(encoding="utf-8")
        if actual != outputs[name]:
            problems.append(f"stale: docs/generated/{name}")

    if problems:
        print("Generated API documentation is out of date:")
        for problem in problems:
            print(f"  {problem}")
        print("Run: python scripts/generate_api_docs.py")
        return False
    return True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Markdown for the declared QAT API."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale generated docs without modifying files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        outputs = render_outputs()
    except (OSError, SyntaxError, ValueError) as error:
        print(f"Unable to generate API documentation: {error}")
        return 2

    if args.check:
        return 0 if check_outputs(outputs) else 1
    write_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
