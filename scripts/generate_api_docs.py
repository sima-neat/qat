#!/usr/bin/env python3
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "generated"
MODULES = [
    ("sima_qat.session", REPO_ROOT / "sima_qat" / "session.py"),
    ("sima_qat.qat_api", REPO_ROOT / "sima_qat" / "qat_api.py"),
    ("sima_qat.sima_quantizer", REPO_ROOT / "sima_qat" / "sima_quantizer.py"),
]


def public_defs(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            yield node


def signature(node):
    if isinstance(node, ast.ClassDef):
        return node.name
    args = [arg.arg for arg in node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwonlyargs:
        args.append("*")
        args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def first_sentence(doc):
    if not doc:
        return ""
    line = " ".join(doc.strip().split())
    if ". " in line:
        return line.split(". ", 1)[0] + "."
    return line


def write_module_doc(module_name: str, source: Path) -> str:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    page_name = module_name.replace(".", "-") + ".md"
    lines = [f"# `{module_name}`", "", f"Source: `{source.relative_to(REPO_ROOT)}`", ""]
    module_doc = ast.get_docstring(tree)
    if module_doc:
        lines.extend([module_doc.strip(), ""])
    defs = list(public_defs(tree))
    if defs:
        lines.extend(["## Public API", ""])
    for node in defs:
        kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
        lines.extend([f"### {kind}: `{signature(node)}`", ""])
        doc = ast.get_docstring(node)
        lines.extend([doc.strip() if doc else "No docstring available.", ""])
    output = "\n".join(lines).rstrip() + "\n"
    output = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    (OUTPUT_DIR / page_name).write_text(output, encoding="utf-8")
    return page_name


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = []
    for module_name, source in MODULES:
        pages.append((module_name, write_module_doc(module_name, source)))

    index = ["# QAT API Reference", "", "Generated from the QAT Python source using `scripts/generate_api_docs.py`.", "", "## Modules", ""]
    for module_name, page in pages:
        tree = ast.parse((REPO_ROOT / module_name.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8"))
        summary = first_sentence(ast.get_docstring(tree))
        suffix = f" - {summary}" if summary else ""
        index.append(f"- [`{module_name}`]({page}){suffix}")
    output = "\n".join(index) + "\n"
    output = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    (OUTPUT_DIR / "index.md").write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
