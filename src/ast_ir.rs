//! Typed IR shared between the Python AST walker (gilmap/_jit.py) and the
//! Cranelift codegen in `jit.rs`. Covers single-`return <expr>` bodies plus
//! multi-statement bodies with local assigns, `for`/`while` loops,
//! `if`/`else` (incl. early return), and `break`/`continue`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    I64,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnaryOp {
    Neg,
    Plus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathFn {
    Sqrt,
    Abs,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)]
pub enum Expr {
    /// Function input (single arg).
    Param,
    /// Local variable read. The walker emits this when it sees a Name
    /// load whose id is a previously-assigned local. Type is recovered
    /// during codegen from the Assign that introduced the local.
    Local { name: String },
    ConstI64 { value: i64 },
    ConstF64 { value: f64 },
    ConstBool { value: bool },
    BinOp { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Unary { op: UnaryOp, operand: Box<Expr> },
    Compare { op: CmpOp, left: Box<Expr>, right: Box<Expr> },
    IfExpr { test: Box<Expr>, yes: Box<Expr>, no: Box<Expr> },
    MathCall { func: MathFn, arg: Box<Expr> },
    /// `int(<expr>)` — explicit cast from f64 to i64. Used by walker when
    /// an `i64` slot is being assigned an `f64`-typed expression.
    CastToI64 { value: Box<Expr> },
    /// `float(<expr>)` — explicit cast from i64 to f64.
    CastToF64 { value: Box<Expr> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Stmt {
    /// `name = value`. `dtype` is inferred by the walker.
    Assign { name: String, dtype: Dtype, value: Expr },
    /// `for var in range(start, end):` — both bounds may be constants or
    /// expressions evaluated once before the loop. v1 single-arg form
    /// emits `start = ConstI64(0)`. Step is fixed at 1; explicit step is
    /// future work.
    ForRange { var: String, start: Expr, end: Expr, body: Vec<Stmt> },
    /// `while cond: body` — re-evaluates `cond` each iteration.
    While { cond: Expr, body: Vec<Stmt> },
    /// `if test: then` or `if test: then; else: orelse`. Both branches
    /// are arbitrary statement lists.
    If { test: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt> },
    /// `return value`.
    Return { value: Expr },
    /// `break` — jumps to the exit block of the innermost enclosing loop.
    Break,
    /// `continue` — jumps to the header block of the innermost enclosing loop.
    Continue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kernel {
    /// Type of the function's input parameter.
    pub input_dtype: Dtype,
    /// Type of the function's return value. May differ from input_dtype
    /// (mandelbrot: input f64, returns i64 iteration count).
    pub output_dtype: Dtype,
    /// Body: either a single Return-of-expression (v1 shape) or a list
    /// of statements (v2). Always serialized as a list.
    pub body: Vec<Stmt>,
}

impl Kernel {
    /// True if this kernel can be lowered into a SIMD vector main loop +
    /// scalar tail (Phase 1 SIMD shape).
    ///
    /// Requirements:
    /// - Body is a single `Stmt::Return`.
    /// - Returned expression has no per-lane divergence:
    ///   - no `Local` references (Phase 1 disables locals to avoid
    ///     plumbing per-lane SSA bookkeeping);
    ///   - no `MathCall` (libm has no SIMD-vectorized variant in
    ///     Cranelift's current symbol table — vectorizing it would
    ///     require lane-extract + scalar call + lane-insert, which is
    ///     slower than the scalar codegen path).
    ///
    /// Anything else (loops, if-statements, locals, math calls) is
    /// emitted by the existing scalar codegen — no regression risk.
    pub fn is_vectorizable_phase1(&self) -> bool {
        if self.body.len() != 1 {
            return false;
        }
        let Stmt::Return { value } = &self.body[0] else {
            return false;
        };
        is_pure_simd_expr(value)
    }
}

fn is_pure_simd_expr(e: &Expr) -> bool {
    match e {
        Expr::Param
        | Expr::ConstI64 { .. }
        | Expr::ConstF64 { .. }
        | Expr::ConstBool { .. } => true,
        Expr::Local { .. } | Expr::MathCall { .. } => false,
        Expr::Unary { operand, .. } => is_pure_simd_expr(operand),
        Expr::BinOp { op, left, right } => {
            // Cranelift's vector backends don't lane-lower these:
            //   - Mod (`srem`): scalar-int only
            //   - FloorDiv: lowered through srem on i64, fmod on f64 (libm)
            //   - Div on i64 (`sdiv`): scalar-int only
            //   - Pow: libm scalar-only
            // f64 Div (`fdiv`) IS vector-friendly but distinguishing
            // op-dtype here would need full type inference; keep the
            // predicate conservative — these cases stay on scalar.
            if matches!(op, BinOp::Mod | BinOp::FloorDiv | BinOp::Div | BinOp::Pow) {
                return false;
            }
            is_pure_simd_expr(left) && is_pure_simd_expr(right)
        }
        Expr::Compare { left, right, .. } => {
            is_pure_simd_expr(left) && is_pure_simd_expr(right)
        }
        Expr::IfExpr { test, yes, no } => {
            is_pure_simd_expr(test) && is_pure_simd_expr(yes) && is_pure_simd_expr(no)
        }
        Expr::CastToI64 { value } | Expr::CastToF64 { value } => is_pure_simd_expr(value),
    }
}
