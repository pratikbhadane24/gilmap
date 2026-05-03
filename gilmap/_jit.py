"""Python AST → typed-IR JSON marshaller for the Cranelift JIT backend.

v1 (P5a): single-`return <expr>` numeric bodies.
v2 (P5b): multi-statement bodies with local `Assign`, counted
`for i in range(N): ...`, `if cond: return X`, and final `return X`.
That's the shape `mandelbrot_iters` needs.

The walker performs lightweight type inference per local variable so that
each `Assign` carries an `i64` or `f64` `dtype` tag in the IR. Inference
rules (intentionally conservative — bail on ambiguity):
  - `int(...)` literal → i64
  - `float(...)` literal → f64
  - `math.X(...)` call → f64
  - Local read → the type recorded when the local was assigned
  - `Param` → kernel input dtype
  - BinOp/Unary/Compare/IfExpr → propagate operand types; if either
    operand is f64, the result is f64; both i64 → i64. (Mod/FloorDiv on
    f64 fail; codegen also rejects.)
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from typing import Callable

from _gilmap import jit_apply, jit_compile


_BINOP_MAP = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
}
_CMP_MAP = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}
_UNARY_MAP = {ast.USub: "neg", ast.UAdd: "plus"}
_MATH_MAP = {
    "sqrt": "sqrt",
    "fabs": "abs",
    "abs": "abs",
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "floor": "floor",
    "ceil": "ceil",
}


class _LowerError(Exception):
    pass


class _Walker:
    def __init__(self, param_name: str, input_dtype: str):
        self.param = param_name
        self.input_dtype = input_dtype
        # name -> "i64" | "f64"
        self.locals: dict[str, str] = {}

    # --- expression lowering + inference -----------------------------

    def expr(self, node: ast.expr) -> tuple[dict, str]:
        """Returns (ir_node, dtype). dtype is "i64" or "f64"."""
        if isinstance(node, ast.Name):
            if node.id == self.param:
                return {"kind": "param"}, self.input_dtype
            if node.id in self.locals:
                return {"kind": "local", "name": node.id}, self.locals[node.id]
            raise _LowerError(f"unknown name '{node.id}' (free var or unassigned)")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return {"kind": "const_bool", "value": node.value}, "i64"
            if isinstance(node.value, int):
                return {"kind": "const_i64", "value": int(node.value)}, "i64"
            if isinstance(node.value, float):
                return {"kind": "const_f64", "value": float(node.value)}, "f64"
            raise _LowerError(f"unsupported constant {type(node.value).__name__}")
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_MAP:
                raise _LowerError(f"unary op {type(node.op).__name__} not supported")
            inner, dt = self.expr(node.operand)
            return {"kind": "unary", "op": _UNARY_MAP[type(node.op)], "operand": inner}, dt
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOP_MAP:
                raise _LowerError(f"binop {type(node.op).__name__} not supported")
            left, lt = self.expr(node.left)
            right, rt = self.expr(node.right)
            # Promotion: if either side is f64, both lift to f64.
            if lt == "f64" or rt == "f64":
                if lt == "i64":
                    left = {"kind": "cast_to_f64", "value": left}
                if rt == "i64":
                    right = {"kind": "cast_to_f64", "value": right}
                result_dt = "f64"
            else:
                result_dt = "i64"
            # FloorDiv/Mod on f64 are not supported by codegen; reject early.
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and result_dt == "f64":
                raise _LowerError("f64 floor/mod not supported in JIT")
            return {
                "kind": "bin_op",
                "op": _BINOP_MAP[type(node.op)],
                "left": left,
                "right": right,
            }, result_dt
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise _LowerError("chained comparisons not supported")
            if type(node.ops[0]) not in _CMP_MAP:
                raise _LowerError("comparison op not supported")
            left, lt = self.expr(node.left)
            right, rt = self.expr(node.comparators[0])
            if lt == "f64" or rt == "f64":
                if lt == "i64":
                    left = {"kind": "cast_to_f64", "value": left}
                if rt == "i64":
                    right = {"kind": "cast_to_f64", "value": right}
            return {
                "kind": "compare",
                "op": _CMP_MAP[type(node.ops[0])],
                "left": left,
                "right": right,
            }, "i64"  # boolean — i8 in Cranelift, but i64 in our IR vocab
        if isinstance(node, ast.IfExp):
            test, _ = self.expr(node.test)
            yes, yt = self.expr(node.body)
            no, nt = self.expr(node.orelse)
            if yt != nt:
                # Promote both to f64 if mixed.
                if yt == "i64":
                    yes = {"kind": "cast_to_f64", "value": yes}
                if nt == "i64":
                    no = {"kind": "cast_to_f64", "value": no}
                rt = "f64"
            else:
                rt = yt
            return {"kind": "if_expr", "test": test, "yes": yes, "no": no}, rt
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr in _MATH_MAP
                and len(node.args) == 1
                and not node.keywords
            ):
                arg, at = self.expr(node.args[0])
                if at == "i64":
                    arg = {"kind": "cast_to_f64", "value": arg}
                return {
                    "kind": "math_call",
                    "func": _MATH_MAP[node.func.attr],
                    "arg": arg,
                }, "f64"
            # int(x) / float(x) explicit casts.
            if isinstance(node.func, ast.Name) and not node.keywords and len(node.args) == 1:
                inner, it = self.expr(node.args[0])
                if node.func.id == "int":
                    if it == "f64":
                        inner = {"kind": "cast_to_i64", "value": inner}
                    return inner, "i64"
                if node.func.id == "float":
                    if it == "i64":
                        inner = {"kind": "cast_to_f64", "value": inner}
                    return inner, "f64"
            raise _LowerError(
                "only math.<sqrt|abs|exp|log|sin|cos|tan|floor|ceil>(arg), int(x), or float(x) calls supported"
            )
        raise _LowerError(f"AST node {type(node).__name__} not supported in JIT")

    # --- statement lowering ------------------------------------------

    def stmts(self, body: list[ast.stmt]) -> tuple[list[dict], bool]:
        """Returns (ir_stmts, terminated). terminated == True if every path
        through the block ends in a Return."""
        # Strip leading docstring.
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        out: list[dict] = []
        terminated = False
        for stmt in body:
            ir, term = self.stmt(stmt)
            out.append(ir)
            if term:
                terminated = True
                break
        return out, terminated

    def stmt(self, node: ast.stmt) -> tuple[dict, bool]:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise _LowerError("only single-name assignments supported")
            name = node.targets[0].id
            value, dt = self.expr(node.value)
            self.locals[name] = dt
            return {"kind": "assign", "name": name, "dtype": dt, "value": value}, False
        if isinstance(node, ast.AugAssign):
            # Lower `x += y` to `x = x + y`.
            if not isinstance(node.target, ast.Name):
                raise _LowerError("augassign target must be a Name")
            name = node.target.id
            if name not in self.locals:
                raise _LowerError(f"augassign on unknown local '{name}'")
            ast_op = node.op
            if type(ast_op) not in _BINOP_MAP:
                raise _LowerError("augassign op not supported")
            left, lt = self.expr(ast.Name(id=name, ctx=ast.Load()))
            right, rt = self.expr(node.value)
            if lt == "f64" or rt == "f64":
                if lt == "i64":
                    left = {"kind": "cast_to_f64", "value": left}
                if rt == "i64":
                    right = {"kind": "cast_to_f64", "value": right}
                rdt = "f64"
            else:
                rdt = "i64"
            value = {
                "kind": "bin_op",
                "op": _BINOP_MAP[type(ast_op)],
                "left": left,
                "right": right,
            }
            # Don't change the local's type — keep what it was.
            existing_dt = self.locals[name]
            if rdt != existing_dt:
                # AugAssign that changes type is an error in our IR.
                raise _LowerError(
                    f"augassign would change type of '{name}' from {existing_dt} to {rdt}"
                )
            return {"kind": "assign", "name": name, "dtype": existing_dt, "value": value}, False
        if isinstance(node, ast.Return):
            if node.value is None:
                raise _LowerError("return with no value not supported")
            value, _dt = self.expr(node.value)
            return {"kind": "return", "value": value}, True
        if isinstance(node, ast.If):
            # Recognize the `if cond: return X` pattern (no else, single
            # Return in body). That's the only If shape v2 supports.
            if (
                node.orelse == []
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Return)
                and node.body[0].value is not None
            ):
                test, _ = self.expr(node.test)
                value, _ = self.expr(node.body[0].value)
                return {"kind": "if_return", "test": test, "value": value}, False
            raise _LowerError(
                "only `if cond: return X` (single-statement, no else) supported in JIT"
            )
        if isinstance(node, ast.For):
            # `for var in range(end):` — only the 1-arg form for now.
            if not isinstance(node.target, ast.Name):
                raise _LowerError("for-loop target must be a Name")
            if node.orelse:
                raise _LowerError("for-loop else clause not supported")
            if not (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and len(node.iter.args) == 1
                and not node.iter.keywords
            ):
                raise _LowerError("only `for var in range(end):` supported")
            end, et = self.expr(node.iter.args[0])
            if et == "f64":
                end = {"kind": "cast_to_i64", "value": end}
            var_name = node.target.id
            self.locals[var_name] = "i64"
            inner_ir, _ = self.stmts(node.body)
            return (
                {"kind": "for_range", "var": var_name, "end": end, "body": inner_ir},
                False,
            )
        raise _LowerError(f"statement {type(node).__name__} not supported in JIT")


def _func_def(func: Callable):
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            return node
    return None


def _infer_output_dtype(stmts: list[dict], input_dtype: str) -> str:
    """Walk the IR looking for Return statements; if any of them yields
    f64, output_dtype is f64; otherwise i64. Conservative — defaults to
    input_dtype when no explicit Return is reachable (defensive)."""
    saw_f64 = False
    saw_any = False
    def visit(stmt: dict):
        nonlocal saw_f64, saw_any
        if stmt.get("kind") == "return":
            saw_any = True
            v = stmt.get("value", {})
            if _expr_is_f64(v):
                saw_f64 = True
        elif stmt.get("kind") == "if_return":
            saw_any = True
            v = stmt.get("value", {})
            if _expr_is_f64(v):
                saw_f64 = True
        elif stmt.get("kind") == "for_range":
            for s in stmt.get("body", []):
                visit(s)
    for s in stmts:
        visit(s)
    if not saw_any:
        return input_dtype
    return "f64" if saw_f64 else "i64"


def _expr_is_f64(expr: dict) -> bool:
    """Best-effort dtype check on an IR expression — only used to decide
    Return type. Mirrors the walker's promotion rules."""
    k = expr.get("kind")
    if k in ("const_f64", "math_call", "cast_to_f64"):
        return True
    if k in ("const_i64", "const_bool", "cast_to_i64"):
        return False
    if k == "bin_op":
        return _expr_is_f64(expr["left"]) or _expr_is_f64(expr["right"])
    if k == "unary":
        return _expr_is_f64(expr["operand"])
    if k == "if_expr":
        return _expr_is_f64(expr["yes"]) or _expr_is_f64(expr["no"])
    if k == "compare":
        return False  # boolean-as-int
    # param/local: we don't track from here. Default to int — consumers
    # that assigned an f64 would already have set output_dtype from a
    # different return path.
    return False


def try_compile(func: Callable, input_dtype: str) -> tuple[int, str] | None:
    """Compile `func` if its body fits the JIT whitelist.

    Returns (kernel_hash, output_dtype) on success, None on rejection.
    `input_dtype` is the dtype of the data being mapped (`"i64"` or `"f64"`).
    """
    fn_node = _func_def(func)
    if fn_node is None:
        return None
    if isinstance(fn_node, ast.Lambda):
        # Lambdas: single-expression body only — fall back to v1 path
        # (single Return).
        params = fn_node.args.args
        if len(params) != 1:
            return None
        walker = _Walker(params[0].arg, input_dtype)
        try:
            value, dt = walker.expr(fn_node.body)
        except _LowerError:
            return None
        body_ir = [{"kind": "return", "value": value}]
        out_dt = dt
    else:
        params = fn_node.args.args
        if len(params) != 1:
            return None
        walker = _Walker(params[0].arg, input_dtype)
        try:
            body_ir, terminated = walker.stmts(fn_node.body)
        except _LowerError:
            return None
        if not terminated:
            # The kernel may exit without an explicit Return — reject so
            # codegen doesn't have to invent a default.
            return None
        out_dt = _infer_output_dtype(body_ir, input_dtype)

    kernel = {"input_dtype": input_dtype, "output_dtype": out_dt, "body": body_ir}
    payload = json.dumps(kernel, sort_keys=True)
    try:
        h = jit_compile(payload)
    except Exception:
        return None
    return h, out_dt


__all__ = ["try_compile", "jit_apply"]
