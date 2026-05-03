"""Shared AST utilities used by both the arrow-kernel and JIT detectors.

Both backends parse the user callable's source, find its FunctionDef or
Lambda node, and lower its body. Centralizing the parse/find logic here
keeps the two backends in sync about what a "valid callable shape" is.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable


def get_source(func: Callable) -> str | None:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    return textwrap.dedent(src)


def find_fn_node(func: Callable) -> ast.FunctionDef | ast.Lambda | None:
    src = get_source(func)
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            return node
    return None


def strip_docstring(stmts: list[ast.stmt]) -> list[ast.stmt]:
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        return stmts[1:]
    return stmts
