"""Python AST → typed-IR JSON marshaller for the Cranelift JIT backend.

The walker handles single-`return <expr>` bodies and multi-statement bodies
with local `Assign`/`AugAssign`, counted `for i in range(...)` loops, `while`
loops, `if`/`else` (incl. `if cond: return X`), `break`, and `continue`.

Type inference is per local: each `Assign` carries an `i64`/`f64` `dtype` tag.
Promotion is conservative — `BinOp`/`Compare`/`IfExpr` lift to f64 when either
operand is f64; otherwise stay i64.
"""

from __future__ import annotations

import ast
import json
from typing import Callable

from _gilmap import jit_apply, jit_compile

from . import _ast_utils


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


def _cast(node: dict, target: str) -> dict:
    return {"kind": f"cast_to_{target}", "value": node}


def _promote(left: dict, lt: str, right: dict, rt: str) -> tuple[dict, dict, str]:
    """Lift mixed-dtype operands to a common type. f64 wins over i64."""
    if lt == rt:
        return left, right, lt
    if lt == "i64":
        left = _cast(left, "f64")
    if rt == "i64":
        right = _cast(right, "f64")
    return left, right, "f64"


class _Walker:
    def __init__(self, param_name: str, input_dtype: str):
        self.param = param_name
        self.input_dtype = input_dtype
        self.locals: dict[str, str] = {}
        # Output dtype is whatever the first reachable Return assigns.
        # Subsequent Returns must agree (codegen requires a single output
        # dtype per kernel).
        self.output_dtype: str | None = None

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
            left, right, result_dt = _promote(left, lt, right, rt)
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
            left, right, _ = _promote(left, lt, right, rt)
            return {
                "kind": "compare",
                "op": _CMP_MAP[type(node.ops[0])],
                "left": left,
                "right": right,
            }, "i64"
        if isinstance(node, ast.IfExp):
            test, _ = self.expr(node.test)
            yes, yt = self.expr(node.body)
            no, nt = self.expr(node.orelse)
            yes, no, rt = _promote(yes, yt, no, nt)
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
                    arg = _cast(arg, "f64")
                return {
                    "kind": "math_call",
                    "func": _MATH_MAP[node.func.attr],
                    "arg": arg,
                }, "f64"
            if isinstance(node.func, ast.Name) and not node.keywords and len(node.args) == 1:
                inner, it = self.expr(node.args[0])
                if node.func.id == "int":
                    return (_cast(inner, "i64") if it == "f64" else inner), "i64"
                if node.func.id == "float":
                    return (_cast(inner, "f64") if it == "i64" else inner), "f64"
            raise _LowerError(
                "only math.<sqrt|abs|exp|log|sin|cos|tan|floor|ceil>(arg), int(x), or float(x) calls supported"
            )
        raise _LowerError(f"AST node {type(node).__name__} not supported in JIT")

    def _record_return(self, dt: str) -> None:
        if self.output_dtype is None:
            self.output_dtype = dt
        elif self.output_dtype != dt:
            raise _LowerError(
                f"return-type mismatch: kernel sets {self.output_dtype} then {dt}"
            )

    def stmts(self, body: list[ast.stmt]) -> tuple[list[dict], bool]:
        """Returns (ir_stmts, terminated)."""
        body = _ast_utils.strip_docstring(body)
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
            if not isinstance(node.target, ast.Name):
                raise _LowerError("augassign target must be a Name")
            name = node.target.id
            if name not in self.locals:
                raise _LowerError(f"augassign on unknown local '{name}'")
            if type(node.op) not in _BINOP_MAP:
                raise _LowerError("augassign op not supported")
            left, lt = self.expr(ast.Name(id=name, ctx=ast.Load()))
            right, rt = self.expr(node.value)
            left, right, rdt = _promote(left, lt, right, rt)
            existing_dt = self.locals[name]
            if rdt != existing_dt:
                raise _LowerError(
                    f"augassign would change type of '{name}' from {existing_dt} to {rdt}"
                )
            value = {
                "kind": "bin_op",
                "op": _BINOP_MAP[type(node.op)],
                "left": left,
                "right": right,
            }
            return {"kind": "assign", "name": name, "dtype": existing_dt, "value": value}, False
        if isinstance(node, ast.Return):
            if node.value is None:
                raise _LowerError("return with no value not supported")
            value, dt = self.expr(node.value)
            self._record_return(dt)
            return {"kind": "return", "value": value}, True
        if isinstance(node, ast.If):
            test, _ = self.expr(node.test)
            then_ir, then_term = self.stmts(node.body)
            else_ir, else_term = self.stmts(node.orelse) if node.orelse else ([], False)
            return (
                {
                    "kind": "if",
                    "test": test,
                    "then_body": then_ir,
                    "else_body": else_ir,
                },
                then_term and else_term,
            )
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise _LowerError("for-loop target must be a Name")
            if node.orelse:
                raise _LowerError("for-loop else clause not supported")
            if not (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and 1 <= len(node.iter.args) <= 2
                and not node.iter.keywords
            ):
                raise _LowerError("only `for var in range(end)` or `range(start, end)` supported")
            args = node.iter.args
            if len(args) == 1:
                start_ir, start_dt = {"kind": "const_i64", "value": 0}, "i64"
                end_ir, end_dt = self.expr(args[0])
            else:
                start_ir, start_dt = self.expr(args[0])
                end_ir, end_dt = self.expr(args[1])
            if start_dt == "f64":
                start_ir = _cast(start_ir, "i64")
            if end_dt == "f64":
                end_ir = _cast(end_ir, "i64")
            var_name = node.target.id
            self.locals[var_name] = "i64"
            inner_ir, _ = self.stmts(node.body)
            return (
                {
                    "kind": "for_range",
                    "var": var_name,
                    "start": start_ir,
                    "end": end_ir,
                    "body": inner_ir,
                },
                False,
            )
        if isinstance(node, ast.While):
            if node.orelse:
                raise _LowerError("while-else not supported")
            cond, _ = self.expr(node.test)
            inner_ir, _ = self.stmts(node.body)
            return ({"kind": "while", "cond": cond, "body": inner_ir}, False)
        if isinstance(node, ast.Break):
            return ({"kind": "break"}, True)
        if isinstance(node, ast.Continue):
            return ({"kind": "continue"}, True)
        raise _LowerError(f"statement {type(node).__name__} not supported in JIT")


def try_compile(func: Callable, input_dtype: str) -> tuple[int, str] | None:
    """Compile `func` if its body fits the JIT whitelist.

    Returns ``(kernel_hash, output_dtype)`` on success, ``None`` on rejection.
    `input_dtype` is the dtype of the data being mapped (`"i64"` or `"f64"`).
    """
    fn_node = _ast_utils.find_fn_node(func)
    if fn_node is None:
        return None
    params = fn_node.args.args
    if len(params) != 1:
        return None
    walker = _Walker(params[0].arg, input_dtype)
    if isinstance(fn_node, ast.Lambda):
        try:
            value, dt = walker.expr(fn_node.body)
        except _LowerError:
            return None
        body_ir = [{"kind": "return", "value": value}]
        out_dt = dt
    else:
        try:
            body_ir, terminated = walker.stmts(fn_node.body)
        except _LowerError:
            return None
        if not terminated:
            return None
        out_dt = walker.output_dtype or input_dtype

    kernel = {"input_dtype": input_dtype, "output_dtype": out_dt, "body": body_ir}
    payload = json.dumps(kernel, sort_keys=True)
    try:
        h = jit_compile(payload)
    except Exception:
        return None
    return h, out_dt


__all__ = ["try_compile", "jit_apply"]
