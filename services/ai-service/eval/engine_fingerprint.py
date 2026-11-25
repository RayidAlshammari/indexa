"""AST fingerprint of the engine modules (docstrings and formatting ignored).

The evaluation ran the engine inside a browser sandbox (Pyodide) because the
build container could not reach the model API. This fingerprint proves the
sandbox ran the same code that is in the repository.

    python eval/engine_fingerprint.py
"""

import ast
import hashlib
from pathlib import Path

MODULES = ["text", "documents", "retrieval", "controls", "judge", "scoring", "engine", "evaluation"]
SRC = Path(__file__).resolve().parents[1] / "src" / "indexa"


def fingerprint(name: str) -> str:
    tree = ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.unparse(tree).encode()).hexdigest()[:10]


if __name__ == "__main__":
    for m in MODULES:
        print(f"{m:<11} {fingerprint(m)}")
