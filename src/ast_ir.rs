//! Typed IR shared between the Python AST walker (gilmap/_jit.py)
//! and the Cranelift codegen in `jit.rs`.
//!
//! v1 (P5a): single-`return <expr>` bodies.
//! v2 (P5b): multi-statement bodies with local assigns, counted
//! `for i in range(N)` loops, and if-statements with early return.
//! That's the shape `mandelbrot_iters` needs.

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
    /// `name = value` — declare or overwrite a local. `dtype` is the type
    /// of the value (inferred by the walker, also used by codegen to
    /// declare a Cranelift Variable on first assignment).
    Assign { name: String, dtype: Dtype, value: Expr },
    /// `for var in range(end): body` — counted loop with i64 counter.
    /// `end` may be a Local read (e.g. `range(max_iter)`) or a constant
    /// expression evaluated once before the loop.
    ForRange { var: String, end: Expr, body: Vec<Stmt> },
    /// `if test: return value` — emitted by the walker for the common
    /// early-return-inside-loop pattern.
    IfReturn { test: Expr, value: Expr },
    /// `return value` — final return at end of body or branch.
    Return { value: Expr },
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
