//! Cranelift JIT.
//!
//! Compiled signature is `extern "C" fn(*const T_in, *mut T_out, len: usize)`.
//! Per element: load input → run body with locals + control flow → store
//! into the output slot at the same index. Early `return` inside the body
//! breaks out of all enclosing loops via a per-element exit block.
//! Supports single-expression bodies plus multi-statement bodies with
//! local assigns, `for`/`while` loops, `if`/`else`, and `break`/`continue`.

use crate::ast_ir::{BinOp, CmpOp, Dtype, Expr, Kernel, MathFn, Stmt, UnaryOp};
use cranelift_codegen::ir::{
    AbiParam, Block, Function, InstBuilder, MemFlags, Signature, Type, Value, types,
};
use cranelift_codegen::isa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub struct Compiled {
    pub fn_ptr: usize,
    _module: JITModule,
}

unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

static KERNEL_REGISTRY: OnceLock<Mutex<HashMap<u64, &'static Compiled>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<u64, &'static Compiled>> {
    KERNEL_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn compile_and_register(kernel: &Kernel, kernel_hash: u64) -> Result<usize, String> {
    {
        let map = registry().lock().unwrap();
        if let Some(c) = map.get(&kernel_hash) {
            return Ok(c.fn_ptr);
        }
    }
    let compiled = compile(kernel)?;
    let leaked: &'static Compiled = Box::leak(Box::new(compiled));
    let fn_ptr = leaked.fn_ptr;
    registry().lock().unwrap().insert(kernel_hash, leaked);
    Ok(fn_ptr)
}

pub fn lookup(kernel_hash: u64) -> Option<usize> {
    registry()
        .lock()
        .unwrap()
        .get(&kernel_hash)
        .map(|c| c.fn_ptr)
}

fn cl_type(dtype: Dtype) -> Type {
    match dtype {
        Dtype::I64 => types::I64,
        Dtype::F64 => types::F64,
    }
}

/// 128-bit SIMD lane type for Phase 1 vectorized JIT codegen.
/// Both i64 and f64 use 2 lanes to keep input/output lane counts matched
/// (so an i64-output kernel from f64 input still vectorizes cleanly).
/// 128-bit is the universally legalized vector width across Cranelift's
/// x86_64 (SSE2+) and aarch64 (NEON) backends — wider lanes (X4/X8) are
/// a separate follow-up gated on ISA feature detection.
const SIMD_LANES: u32 = 2;

fn cl_vec_type(dtype: Dtype) -> Type {
    match dtype {
        Dtype::I64 => types::I64X2,
        Dtype::F64 => types::F64X2,
    }
}

/// Build a JIT module with all the boilerplate: ISA flags, libm symbols,
/// the standard `extern "C" fn(*const u8, *mut u8, usize)` signature.
/// Both the scalar and SIMD compile paths start from this.
fn setup_jit_module() -> Result<(JITModule, Type, Signature), String> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| e.to_string())?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| e.to_string())?;
    let isa_builder = isa::lookup(target_lexicon::Triple::host())
        .map_err(|e| format!("ISA lookup failed: {}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| format!("ISA finish failed: {}", e))?;

    let mut jit_builder =
        JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_libm_symbols(&mut jit_builder);
    let module = JITModule::new(jit_builder);
    let pointer_ty = module.target_config().pointer_type();

    let mut sig = Signature::new(module.target_config().default_call_conv);
    sig.params.push(AbiParam::new(pointer_ty));
    sig.params.push(AbiParam::new(pointer_ty));
    sig.params.push(AbiParam::new(pointer_ty));

    Ok((module, pointer_ty, sig))
}

fn compile(kernel: &Kernel) -> Result<Compiled, String> {
    if kernel.is_vectorizable_phase1() {
        return compile_simd(kernel);
    }
    compile_scalar(kernel)
}

fn compile_scalar(kernel: &Kernel) -> Result<Compiled, String> {
    let (mut module, pointer_ty, sig) = setup_jit_module()?;
    let in_ty = cl_type(kernel.input_dtype);
    let out_ty = cl_type(kernel.output_dtype);

    let func_id = module
        .declare_function("gilmap_kernel", Linkage::Export, &sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func = Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, 0),
        sig,
    );

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        let outer_header = builder.create_block();
        let elem_entry = builder.create_block();
        let elem_exit = builder.create_block();
        let outer_exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        let in_ptr = builder.block_params(entry)[0];
        let out_ptr = builder.block_params(entry)[1];
        let len = builder.block_params(entry)[2];

        // Outer loop counter (one per element).
        let i_var = builder.declare_var(pointer_ty);
        // Per-element result slot — every Return path writes here, then
        // jumps to elem_exit which stores into out[i].
        let result_var = builder.declare_var(out_ty);

        builder.switch_to_block(entry);
        let zero = builder.ins().iconst(pointer_ty, 0);
        builder.def_var(i_var, zero);
        builder.ins().jump(outer_header, &[]);

        builder.switch_to_block(outer_header);
        let i_now = builder.use_var(i_var);
        let cond = builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
            i_now,
            len,
        );
        builder.ins().brif(cond, elem_entry, &[], outer_exit, &[]);

        builder.switch_to_block(elem_entry);
        // Load input element and bind as Param.
        let elem_size_in = builder.ins().iconst(pointer_ty, in_ty.bytes() as i64);
        let i_now2 = builder.use_var(i_var);
        let off_in = builder.ins().imul(i_now2, elem_size_in);
        let in_addr = builder.ins().iadd(in_ptr, off_in);
        let mem = MemFlags::new();
        let val = builder.ins().load(in_ty, mem, in_addr, 0);

        // Lower body. The Lowerer maintains a HashMap of locals
        // (name → (Variable, Dtype)) and routes Return statements to
        // `elem_exit` after writing into `result_var`.
        let mut lowerer = Lowerer {
            builder: &mut builder,
            module: &mut module,
            input_dtype: kernel.input_dtype,
            output_dtype: kernel.output_dtype,
            param: val,
            locals: HashMap::new(),
            result_var,
            elem_exit,
            terminated: false,
            loops: Vec::new(),
        };
        lowerer.lower_body(&kernel.body)?;

        // If body lowering didn't terminate with a return on every path,
        // the unreachable branch needs *some* terminator. Default: write
        // zero and jump to elem_exit. (Walker requires terminating
        // Return on every path, so this is defensive.)
        if !lowerer.terminated {
            let z = match kernel.output_dtype {
                Dtype::I64 => builder.ins().iconst(out_ty, 0),
                Dtype::F64 => builder.ins().f64const(0.0),
            };
            builder.def_var(result_var, z);
            builder.ins().jump(elem_exit, &[]);
        }

        builder.switch_to_block(elem_exit);
        let result_val = builder.use_var(result_var);
        let elem_size_out = builder.ins().iconst(pointer_ty, out_ty.bytes() as i64);
        let i_now3 = builder.use_var(i_var);
        let off_out = builder.ins().imul(i_now3, elem_size_out);
        let out_addr = builder.ins().iadd(out_ptr, off_out);
        builder.ins().store(mem, result_val, out_addr, 0);
        let one = builder.ins().iconst(pointer_ty, 1);
        let i_next = builder.ins().iadd(i_now3, one);
        builder.def_var(i_var, i_next);
        builder.ins().jump(outer_header, &[]);

        builder.switch_to_block(outer_exit);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define_function: {:?}", e))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {:?}", e))?;

    let fn_ptr = module.get_finalized_function(func_id) as usize;
    Ok(Compiled {
        fn_ptr,
        _module: module,
    })
}

/// SIMD codegen path. Emits two loops over the buffer:
/// 1. Vector main loop: process `SIMD_LANES` elements per iteration via
///    Cranelift vector-typed loads/ops/stores. Cranelift legalizes
///    128-bit operations to native SSE2/NEON instructions on the host.
/// 2. Scalar tail: handle the trailing `len % SIMD_LANES` elements with
///    the same per-element codegen the scalar path uses (via `Lowerer`).
///
/// The scalar tail reuses `Lowerer` rather than inlining a third copy of
/// expression lowering — that way bug-fixes in the scalar path
/// automatically apply here.
fn compile_simd(kernel: &Kernel) -> Result<Compiled, String> {
    // Predicate guarantees body is `[Stmt::Return(expr)]`.
    let Stmt::Return { value: return_expr } = &kernel.body[0] else {
        unreachable!("compile_simd called on non-vectorizable kernel");
    };

    let (mut module, pointer_ty, sig) = setup_jit_module()?;
    let in_ty = cl_type(kernel.input_dtype);
    let out_ty = cl_type(kernel.output_dtype);
    let in_vec_ty = cl_vec_type(kernel.input_dtype);

    let func_id = module
        .declare_function("gilmap_kernel", Linkage::Export, &sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func = Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, 0),
        sig,
    );

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        let vec_header = builder.create_block();
        let vec_body = builder.create_block();
        let scalar_header = builder.create_block();
        let scalar_entry = builder.create_block();
        let scalar_exit = builder.create_block();
        let outer_exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        let in_ptr = builder.block_params(entry)[0];
        let out_ptr = builder.block_params(entry)[1];
        let len = builder.block_params(entry)[2];

        let i_var = builder.declare_var(pointer_ty);
        let tail_result_var = builder.declare_var(out_ty);

        builder.switch_to_block(entry);
        let zero = builder.ins().iconst(pointer_ty, 0);
        builder.def_var(i_var, zero);
        let lanes_const = builder.ins().iconst(pointer_ty, SIMD_LANES as i64);
        // vec_end = len - (len % LANES); cheaper than (len / LANES) * LANES
        // because cranelift can't const-fold the divide on a runtime `len`.
        let len_mod = builder.ins().urem(len, lanes_const);
        let vec_end = builder.ins().isub(len, len_mod);
        let elem_size_in = builder.ins().iconst(pointer_ty, in_ty.bytes() as i64);
        let elem_size_out = builder.ins().iconst(pointer_ty, out_ty.bytes() as i64);
        let one = builder.ins().iconst(pointer_ty, 1);
        let mem = MemFlags::new();
        builder.ins().jump(vec_header, &[]);

        builder.switch_to_block(vec_header);
        let i_now = builder.use_var(i_var);
        let vec_cond = builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
            i_now,
            vec_end,
        );
        builder
            .ins()
            .brif(vec_cond, vec_body, &[], scalar_header, &[]);

        builder.switch_to_block(vec_body);
        let i_now2 = builder.use_var(i_var);
        let off_in = builder.ins().imul(i_now2, elem_size_in);
        let in_addr = builder.ins().iadd(in_ptr, off_in);
        let vec_val = builder.ins().load(in_vec_ty, mem, in_addr, 0);

        let mut simd = SimdLowerer {
            builder: &mut builder,
            input_dtype: kernel.input_dtype,
            param: vec_val,
        };
        let vec_result = simd.lower(return_expr, kernel.output_dtype)?;

        let i_now3 = builder.use_var(i_var);
        let off_out = builder.ins().imul(i_now3, elem_size_out);
        let out_addr = builder.ins().iadd(out_ptr, off_out);
        builder.ins().store(mem, vec_result, out_addr, 0);

        let i_next_vec = builder.ins().iadd(i_now3, lanes_const);
        builder.def_var(i_var, i_next_vec);
        builder.ins().jump(vec_header, &[]);

        builder.switch_to_block(scalar_header);
        let i_tail = builder.use_var(i_var);
        let tail_cond = builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
            i_tail,
            len,
        );
        builder
            .ins()
            .brif(tail_cond, scalar_entry, &[], outer_exit, &[]);

        builder.switch_to_block(scalar_entry);
        let i_s = builder.use_var(i_var);
        let off_in_s = builder.ins().imul(i_s, elem_size_in);
        let in_addr_s = builder.ins().iadd(in_ptr, off_in_s);
        let val_s = builder.ins().load(in_ty, mem, in_addr_s, 0);

        let mut lowerer = Lowerer {
            builder: &mut builder,
            module: &mut module,
            input_dtype: kernel.input_dtype,
            output_dtype: kernel.output_dtype,
            param: val_s,
            locals: HashMap::new(),
            result_var: tail_result_var,
            elem_exit: scalar_exit,
            terminated: false,
            loops: Vec::new(),
        };
        lowerer.lower_body(&kernel.body)?;
        debug_assert!(
            lowerer.terminated,
            "Phase 1 vectorizable bodies always terminate via Return"
        );

        builder.switch_to_block(scalar_exit);
        let result_val = builder.use_var(tail_result_var);
        let i_s2 = builder.use_var(i_var);
        let off_out_s = builder.ins().imul(i_s2, elem_size_out);
        let out_addr_s = builder.ins().iadd(out_ptr, off_out_s);
        builder.ins().store(mem, result_val, out_addr_s, 0);
        let i_next_s = builder.ins().iadd(i_s2, one);
        builder.def_var(i_var, i_next_s);
        builder.ins().jump(scalar_header, &[]);

        builder.switch_to_block(outer_exit);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define_function: {:?}", e))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {:?}", e))?;

    let fn_ptr = module.get_finalized_function(func_id) as usize;
    Ok(Compiled {
        fn_ptr,
        _module: module,
    })
}

/// Vector-typed analogue of `Lowerer` for the SIMD main loop. Phase 1 only
/// — pure expressions, no locals, no MathCall (see
/// `Kernel::is_vectorizable_phase1`).
struct SimdLowerer<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    input_dtype: Dtype,
    param: Value,
}

impl<'a, 'b> SimdLowerer<'a, 'b> {
    fn lower(&mut self, expr: &Expr, expected: Dtype) -> Result<Value, String> {
        Ok(match expr {
            Expr::Param => self.coerce(self.param, self.input_dtype, expected),
            Expr::ConstI64 { value } => {
                let s = self.builder.ins().iconst(types::I64, *value);
                let v = self.builder.ins().splat(types::I64X2, s);
                self.coerce(v, Dtype::I64, expected)
            }
            Expr::ConstF64 { value } => {
                let s = self.builder.ins().f64const(*value);
                let v = self.builder.ins().splat(types::F64X2, s);
                self.coerce(v, Dtype::F64, expected)
            }
            Expr::ConstBool { value } => {
                // All-ones / all-zeros mask shape required by bitselect.
                let s = self.builder.ins().iconst(types::I64, if *value { -1 } else { 0 });
                self.builder.ins().splat(types::I64X2, s)
            }
            Expr::Unary { op, operand } => {
                let v = self.lower(operand, expected)?;
                match (op, expected) {
                    (UnaryOp::Neg, Dtype::I64) => self.builder.ins().ineg(v),
                    (UnaryOp::Neg, Dtype::F64) => self.builder.ins().fneg(v),
                    (UnaryOp::Plus, _) => v,
                }
            }
            Expr::BinOp { op, left, right } => {
                let l = self.lower(left, expected)?;
                let r = self.lower(right, expected)?;
                match (op, expected) {
                    (BinOp::Add, Dtype::I64) => self.builder.ins().iadd(l, r),
                    (BinOp::Sub, Dtype::I64) => self.builder.ins().isub(l, r),
                    (BinOp::Mul, Dtype::I64) => self.builder.ins().imul(l, r),
                    (BinOp::Div, Dtype::I64) | (BinOp::FloorDiv, Dtype::I64) => {
                        self.builder.ins().sdiv(l, r)
                    }
                    (BinOp::Mod, Dtype::I64) => self.builder.ins().srem(l, r),
                    (BinOp::Pow, Dtype::I64) => {
                        return Err("integer pow not supported in SIMD JIT".into());
                    }
                    (BinOp::Add, Dtype::F64) => self.builder.ins().fadd(l, r),
                    (BinOp::Sub, Dtype::F64) => self.builder.ins().fsub(l, r),
                    (BinOp::Mul, Dtype::F64) => self.builder.ins().fmul(l, r),
                    (BinOp::Div, Dtype::F64) => self.builder.ins().fdiv(l, r),
                    (BinOp::FloorDiv, Dtype::F64) | (BinOp::Mod, Dtype::F64) => {
                        return Err("floor/mod on f64 not supported in SIMD JIT".into());
                    }
                    (BinOp::Pow, Dtype::F64) => {
                        return Err(
                            "f64 pow uses libm; not supported in SIMD path".into(),
                        );
                    }
                }
            }
            Expr::Compare { op, left, right } => {
                let cmp_dtype = guess_simd_cmp_dtype(left, right, self.input_dtype);
                let l = self.lower(left, cmp_dtype)?;
                let r = self.lower(right, cmp_dtype)?;
                match cmp_dtype {
                    Dtype::I64 => {
                        use cranelift_codegen::ir::condcodes::IntCC::*;
                        let cc = match op {
                            CmpOp::Lt => SignedLessThan,
                            CmpOp::Le => SignedLessThanOrEqual,
                            CmpOp::Gt => SignedGreaterThan,
                            CmpOp::Ge => SignedGreaterThanOrEqual,
                            CmpOp::Eq => Equal,
                            CmpOp::Ne => NotEqual,
                        };
                        self.builder.ins().icmp(cc, l, r)
                    }
                    Dtype::F64 => {
                        use cranelift_codegen::ir::condcodes::FloatCC::*;
                        let cc = match op {
                            CmpOp::Lt => LessThan,
                            CmpOp::Le => LessThanOrEqual,
                            CmpOp::Gt => GreaterThan,
                            CmpOp::Ge => GreaterThanOrEqual,
                            CmpOp::Eq => Equal,
                            CmpOp::Ne => NotEqual,
                        };
                        self.builder.ins().fcmp(cc, l, r)
                    }
                }
            }
            Expr::IfExpr { test, yes, no } => {
                // bitselect requires all three operands the same type;
                // icmp/fcmp on vectors return an int mask, so reinterpret
                // the i64x2 mask bits as f64x2 when the operands are f64.
                let cond_raw = self.lower(test, Dtype::I64)?;
                let yes_v = self.lower(yes, expected)?;
                let no_v = self.lower(no, expected)?;
                let cond_for_select = match expected {
                    Dtype::I64 => cond_raw,
                    Dtype::F64 => self.builder.ins().bitcast(
                        types::F64X2,
                        MemFlags::new(),
                        cond_raw,
                    ),
                };
                self.builder.ins().bitselect(cond_for_select, yes_v, no_v)
            }
            Expr::CastToI64 { value } => {
                let v = self.lower(value, Dtype::F64)?;
                self.builder.ins().fcvt_to_sint(types::I64X2, v)
            }
            Expr::CastToF64 { value } => {
                let v = self.lower(value, Dtype::I64)?;
                self.builder.ins().fcvt_from_sint(types::F64X2, v)
            }
            Expr::Local { .. } | Expr::MathCall { .. } => {
                unreachable!("predicate rejects Local / MathCall before SIMD lowering")
            }
        })
    }

    fn coerce(&mut self, v: Value, actual: Dtype, expected: Dtype) -> Value {
        match (actual, expected) {
            (Dtype::I64, Dtype::I64) | (Dtype::F64, Dtype::F64) => v,
            (Dtype::I64, Dtype::F64) => self.builder.ins().fcvt_from_sint(types::F64X2, v),
            (Dtype::F64, Dtype::I64) => self.builder.ins().fcvt_to_sint(types::I64X2, v),
        }
    }
}

fn guess_simd_cmp_dtype(left: &Expr, right: &Expr, input_dtype: Dtype) -> Dtype {
    fn inspect(e: &Expr, input_dtype: Dtype) -> Option<Dtype> {
        match e {
            Expr::ConstF64 { .. } | Expr::CastToF64 { .. } => Some(Dtype::F64),
            Expr::ConstI64 { .. } | Expr::CastToI64 { .. } => Some(Dtype::I64),
            Expr::Param => Some(input_dtype),
            Expr::BinOp { left, right, .. } | Expr::Compare { left, right, .. } => {
                inspect(left, input_dtype).or_else(|| inspect(right, input_dtype))
            }
            Expr::Unary { operand, .. } => inspect(operand, input_dtype),
            Expr::IfExpr { yes, no, .. } => {
                inspect(yes, input_dtype).or_else(|| inspect(no, input_dtype))
            }
            Expr::ConstBool { .. } => Some(Dtype::I64),
            Expr::Local { .. } | Expr::MathCall { .. } => None,
        }
    }
    inspect(left, input_dtype)
        .or_else(|| inspect(right, input_dtype))
        .unwrap_or(Dtype::I64)
}

/// Per-loop context for `break`/`continue` targeting. `header_block` is
/// the block control-flow returns to each iteration. For for-range loops
/// it's a synthetic increment block that bumps the counter and re-checks
/// bounds; for while loops it's the cond-eval block.
#[derive(Clone, Copy)]
struct LoopFrame {
    header_block: Block,
    exit_block: Block,
}

struct Lowerer<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    module: &'a mut JITModule,
    input_dtype: Dtype,
    output_dtype: Dtype,
    param: Value,
    locals: HashMap<String, (Variable, Dtype)>,
    result_var: Variable,
    elem_exit: Block,
    /// True when the current block has been terminated (jump/brif/return).
    /// Reset by callers when they switch to a fresh block.
    terminated: bool,
    /// Stack of enclosing loops; innermost is the last entry. break/continue
    /// target the innermost frame.
    loops: Vec<LoopFrame>,
}

impl<'a, 'b> Lowerer<'a, 'b> {
    fn lower_body(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for stmt in stmts {
            if self.terminated {
                break;
            }
            self.lower_stmt(stmt)?;
        }
        Ok(())
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Assign { name, dtype, value } => {
                let v = self.lower_expr(value, *dtype)?;
                let var = if let Some((existing, existing_dtype)) = self.locals.get(name) {
                    if *existing_dtype != *dtype {
                        return Err(format!(
                            "local '{}' redeclared with different dtype: {:?} -> {:?}",
                            name, existing_dtype, dtype
                        ));
                    }
                    *existing
                } else {
                    let new_var = self.builder.declare_var(cl_type(*dtype));
                    self.locals.insert(name.clone(), (new_var, *dtype));
                    new_var
                };
                self.builder.def_var(var, v);
            }
            Stmt::Return { value } => {
                let v = self.lower_expr(value, self.output_dtype)?;
                self.builder.def_var(self.result_var, v);
                self.builder.ins().jump(self.elem_exit, &[]);
                self.terminated = true;
            }
            Stmt::ForRange { var, start, end, body } => {
                let start_val = self.lower_expr(start, Dtype::I64)?;
                let end_val = self.lower_expr(end, Dtype::I64)?;
                let counter = if let Some((existing, _)) = self.locals.get(var) {
                    *existing
                } else {
                    let new_var = self.builder.declare_var(types::I64);
                    self.locals.insert(var.clone(), (new_var, Dtype::I64));
                    new_var
                };
                self.builder.def_var(counter, start_val);

                let header = self.builder.create_block();
                let body_block = self.builder.create_block();
                let exit = self.builder.create_block();

                self.builder.ins().jump(header, &[]);

                self.builder.switch_to_block(header);
                self.terminated = false;
                let i_now = self.builder.use_var(counter);
                let cont = self.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                    i_now,
                    end_val,
                );
                self.builder.ins().brif(cont, body_block, &[], exit, &[]);

                // `continue` inside the body needs to bump the counter
                // before re-checking the bound. Route it through a synthetic
                // increment block so the LoopFrame's header_block already
                // advances the counter.
                let inc_block = self.builder.create_block();
                self.builder.switch_to_block(inc_block);
                self.terminated = false;
                let i_now = self.builder.use_var(counter);
                let one = self.builder.ins().iconst(types::I64, 1);
                let i_next = self.builder.ins().iadd(i_now, one);
                self.builder.def_var(counter, i_next);
                self.builder.ins().jump(header, &[]);

                self.builder.switch_to_block(body_block);
                self.terminated = false;
                self.loops.push(LoopFrame {
                    header_block: inc_block,
                    exit_block: exit,
                });
                self.lower_body(body)?;
                self.loops.pop();
                if !self.terminated {
                    self.builder.ins().jump(inc_block, &[]);
                }

                self.builder.switch_to_block(exit);
                self.terminated = false;
            }
            Stmt::While { cond, body } => {
                let header = self.builder.create_block();
                let body_block = self.builder.create_block();
                let exit = self.builder.create_block();

                self.builder.ins().jump(header, &[]);

                self.builder.switch_to_block(header);
                self.terminated = false;
                let cond_val = self.lower_expr(cond, Dtype::I64)?;
                self.builder.ins().brif(cond_val, body_block, &[], exit, &[]);

                self.builder.switch_to_block(body_block);
                self.terminated = false;
                self.loops.push(LoopFrame {
                    header_block: header,
                    exit_block: exit,
                });
                self.lower_body(body)?;
                self.loops.pop();
                if !self.terminated {
                    self.builder.ins().jump(header, &[]);
                }

                self.builder.switch_to_block(exit);
                self.terminated = false;
            }
            Stmt::If { test, then_body, else_body } => {
                let cond = self.lower_expr(test, Dtype::I64)?;
                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let cont_block = self.builder.create_block();
                self.builder
                    .ins()
                    .brif(cond, then_block, &[], else_block, &[]);

                self.builder.switch_to_block(then_block);
                self.terminated = false;
                self.lower_body(then_body)?;
                if !self.terminated {
                    self.builder.ins().jump(cont_block, &[]);
                }

                self.builder.switch_to_block(else_block);
                self.terminated = false;
                self.lower_body(else_body)?;
                if !self.terminated {
                    self.builder.ins().jump(cont_block, &[]);
                }

                self.builder.switch_to_block(cont_block);
                self.terminated = false;
            }
            Stmt::Break => {
                let frame = self
                    .loops
                    .last()
                    .ok_or("`break` outside of any loop".to_string())?;
                let exit = frame.exit_block;
                self.builder.ins().jump(exit, &[]);
                self.terminated = true;
            }
            Stmt::Continue => {
                let frame = self
                    .loops
                    .last()
                    .ok_or("`continue` outside of any loop".to_string())?;
                self.builder.ins().jump(frame.header_block, &[]);
                self.terminated = true;
            }
        }
        Ok(())
    }

    /// Lower an expression; `expected` is the dtype the consumer wants —
    /// used to insert implicit i64↔f64 promotions for literals so users
    /// can write `cr = x / 1000.0` (mixed) without explicit casts in the
    /// common cases. (For non-literal mixed-type ops the walker should
    /// have inserted explicit `Cast` IR nodes.)
    fn lower_expr(&mut self, expr: &Expr, expected: Dtype) -> Result<Value, String> {
        Ok(match expr {
            Expr::Param => {
                self.coerce(self.param, self.input_dtype, expected)
            }
            Expr::Local { name } => {
                let (var, dtype) = *self
                    .locals
                    .get(name)
                    .ok_or_else(|| format!("unknown local '{}'", name))?;
                let v = self.builder.use_var(var);
                self.coerce(v, dtype, expected)
            }
            Expr::ConstI64 { value } => {
                let v = self.builder.ins().iconst(types::I64, *value);
                self.coerce(v, Dtype::I64, expected)
            }
            Expr::ConstF64 { value } => {
                let v = self.builder.ins().f64const(*value);
                self.coerce(v, Dtype::F64, expected)
            }
            Expr::ConstBool { value } => {
                // Booleans are stored as i64 across the JIT to keep dtype
                // bookkeeping uniform with other ints. Codegen for `brif`
                // accepts any integer type.
                self.builder.ins().iconst(types::I64, if *value { 1 } else { 0 })
            }
            Expr::Unary { op, operand } => {
                let v = self.lower_expr(operand, expected)?;
                match (op, expected) {
                    (UnaryOp::Neg, Dtype::I64) => self.builder.ins().ineg(v),
                    (UnaryOp::Neg, Dtype::F64) => self.builder.ins().fneg(v),
                    (UnaryOp::Plus, _) => v,
                }
            }
            Expr::BinOp { op, left, right } => {
                // For arithmetic, both operands take `expected` dtype. The
                // walker is responsible for inserting CastToF64/CastToI64
                // when operands disagree.
                let l = self.lower_expr(left, expected)?;
                let r = self.lower_expr(right, expected)?;
                self.bin_op(*op, l, r, expected)?
            }
            Expr::Compare { op, left, right } => {
                // Compare is dtype-aware: both operands must match.
                // Use the operand's expected dtype — the test caller will
                // usually pass `Dtype::I64` (since brif takes int) but the
                // operands themselves may be f64. Resolve from the AST:
                // if either side is a float-shaped literal, promote both.
                let cmp_dtype = guess_cmp_dtype(left, right, self);
                let l = self.lower_expr(left, cmp_dtype)?;
                let r = self.lower_expr(right, cmp_dtype)?;
                self.cmp_op(*op, l, r, cmp_dtype)
            }
            Expr::IfExpr { test, yes, no } => {
                let cond = self.lower_expr(test, Dtype::I64)?;
                let yes_v = self.lower_expr(yes, expected)?;
                let no_v = self.lower_expr(no, expected)?;
                self.builder.ins().select(cond, yes_v, no_v)
            }
            Expr::MathCall { func, arg } => {
                if expected != Dtype::F64 {
                    return Err("math.* result is f64; assign to an f64 slot".into());
                }
                let v = self.lower_expr(arg, Dtype::F64)?;
                let name = match func {
                    MathFn::Sqrt => "sqrt",
                    MathFn::Abs => "fabs",
                    MathFn::Exp => "exp",
                    MathFn::Log => "log",
                    MathFn::Sin => "sin",
                    MathFn::Cos => "cos",
                    MathFn::Tan => "tan",
                    MathFn::Floor => "floor",
                    MathFn::Ceil => "ceil",
                };
                let id = libm_call(self.module, name, 1, types::F64)?;
                let local = self.module.declare_func_in_func(id, self.builder.func);
                let inst = self.builder.ins().call(local, &[v]);
                self.builder.inst_results(inst)[0]
            }
            Expr::CastToI64 { value } => {
                let v = self.lower_expr(value, Dtype::F64)?;
                self.builder.ins().fcvt_to_sint(types::I64, v)
            }
            Expr::CastToF64 { value } => {
                let v = self.lower_expr(value, Dtype::I64)?;
                self.builder.ins().fcvt_from_sint(types::F64, v)
            }
        })
    }

    /// If `actual` doesn't match `expected`, insert a cast. Used for
    /// literal/local promotions when the walker hasn't inserted explicit
    /// CastTo* nodes. Two cases: i64→f64 (sint to f64), f64→i64 (truncate).
    fn coerce(&mut self, v: Value, actual: Dtype, expected: Dtype) -> Value {
        match (actual, expected) {
            (Dtype::I64, Dtype::I64) | (Dtype::F64, Dtype::F64) => v,
            (Dtype::I64, Dtype::F64) => self.builder.ins().fcvt_from_sint(types::F64, v),
            (Dtype::F64, Dtype::I64) => self.builder.ins().fcvt_to_sint(types::I64, v),
        }
    }

    fn bin_op(&mut self, op: BinOp, l: Value, r: Value, dtype: Dtype) -> Result<Value, String> {
        Ok(match (op, dtype) {
            (BinOp::Add, Dtype::I64) => self.builder.ins().iadd(l, r),
            (BinOp::Sub, Dtype::I64) => self.builder.ins().isub(l, r),
            (BinOp::Mul, Dtype::I64) => self.builder.ins().imul(l, r),
            (BinOp::Div, Dtype::I64) | (BinOp::FloorDiv, Dtype::I64) => {
                self.builder.ins().sdiv(l, r)
            }
            (BinOp::Mod, Dtype::I64) => self.builder.ins().srem(l, r),
            (BinOp::Pow, Dtype::I64) => return Err("integer pow not supported in JIT".into()),
            (BinOp::Add, Dtype::F64) => self.builder.ins().fadd(l, r),
            (BinOp::Sub, Dtype::F64) => self.builder.ins().fsub(l, r),
            (BinOp::Mul, Dtype::F64) => self.builder.ins().fmul(l, r),
            (BinOp::Div, Dtype::F64) => self.builder.ins().fdiv(l, r),
            (BinOp::FloorDiv, Dtype::F64) | (BinOp::Mod, Dtype::F64) => {
                return Err("floor/mod on f64 not supported in JIT".into());
            }
            (BinOp::Pow, Dtype::F64) => {
                let id = libm_call(self.module, "pow", 2, types::F64)?;
                let local = self.module.declare_func_in_func(id, self.builder.func);
                let inst = self.builder.ins().call(local, &[l, r]);
                self.builder.inst_results(inst)[0]
            }
        })
    }

    fn cmp_op(&mut self, op: CmpOp, l: Value, r: Value, dtype: Dtype) -> Value {
        match dtype {
            Dtype::I64 => {
                use cranelift_codegen::ir::condcodes::IntCC::*;
                let cc = match op {
                    CmpOp::Lt => SignedLessThan,
                    CmpOp::Le => SignedLessThanOrEqual,
                    CmpOp::Gt => SignedGreaterThan,
                    CmpOp::Ge => SignedGreaterThanOrEqual,
                    CmpOp::Eq => Equal,
                    CmpOp::Ne => NotEqual,
                };
                self.builder.ins().icmp(cc, l, r)
            }
            Dtype::F64 => {
                use cranelift_codegen::ir::condcodes::FloatCC::*;
                let cc = match op {
                    CmpOp::Lt => LessThan,
                    CmpOp::Le => LessThanOrEqual,
                    CmpOp::Gt => GreaterThan,
                    CmpOp::Ge => GreaterThanOrEqual,
                    CmpOp::Eq => Equal,
                    CmpOp::Ne => NotEqual,
                };
                self.builder.ins().fcmp(cc, l, r)
            }
        }
    }
}

/// Heuristic for resolving the dtype of a Compare's operands. If either
/// side syntactically requires f64 (Local of f64 type, ConstF64, MathCall),
/// the comparison is f64; otherwise i64. This keeps the walker simpler
/// (it doesn't need to annotate Compare with operand dtype).
fn guess_cmp_dtype(left: &Expr, right: &Expr, l: &Lowerer) -> Dtype {
    fn inspect(e: &Expr, l: &Lowerer) -> Option<Dtype> {
        match e {
            Expr::ConstF64 { .. } | Expr::MathCall { .. } | Expr::CastToF64 { .. } => {
                Some(Dtype::F64)
            }
            Expr::ConstI64 { .. } | Expr::CastToI64 { .. } => Some(Dtype::I64),
            Expr::Param => Some(l.input_dtype),
            Expr::Local { name } => l.locals.get(name).map(|(_, d)| *d),
            Expr::BinOp { left, right, .. } | Expr::Compare { left, right, .. } => {
                inspect(left, l).or_else(|| inspect(right, l))
            }
            Expr::Unary { operand, .. } => inspect(operand, l),
            Expr::IfExpr { yes, no, .. } => inspect(yes, l).or_else(|| inspect(no, l)),
            Expr::ConstBool { .. } => Some(Dtype::I64),
        }
    }
    inspect(left, l).or_else(|| inspect(right, l)).unwrap_or(Dtype::I64)
}

fn libm_call(
    module: &mut JITModule,
    name: &str,
    arity: usize,
    ty: Type,
) -> Result<cranelift_module::FuncId, String> {
    let mut sig = module.make_signature();
    for _ in 0..arity {
        sig.params.push(AbiParam::new(ty));
    }
    sig.returns.push(AbiParam::new(ty));
    module
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| e.to_string())
}

fn register_libm_symbols(builder: &mut JITBuilder) {
    use std::os::raw::c_double;
    unsafe extern "C" {
        fn sqrt(x: c_double) -> c_double;
        fn fabs(x: c_double) -> c_double;
        fn exp(x: c_double) -> c_double;
        fn log(x: c_double) -> c_double;
        fn sin(x: c_double) -> c_double;
        fn cos(x: c_double) -> c_double;
        fn tan(x: c_double) -> c_double;
        fn floor(x: c_double) -> c_double;
        fn ceil(x: c_double) -> c_double;
        fn pow(x: c_double, y: c_double) -> c_double;
    }
    builder.symbol("sqrt", sqrt as *const u8);
    builder.symbol("fabs", fabs as *const u8);
    builder.symbol("exp", exp as *const u8);
    builder.symbol("log", log as *const u8);
    builder.symbol("sin", sin as *const u8);
    builder.symbol("cos", cos as *const u8);
    builder.symbol("tan", tan as *const u8);
    builder.symbol("floor", floor as *const u8);
    builder.symbol("ceil", ceil as *const u8);
    builder.symbol("pow", pow as *const u8);
}

type KernelFn = unsafe extern "C" fn(*const u8, *mut u8, usize);

#[inline]
unsafe fn as_kernel_fn(fn_ptr: usize) -> KernelFn {
    unsafe { std::mem::transmute(fn_ptr) }
}

/// Invoke a kernel in parallel by splitting the input buffer into
/// `num_threads * 4` chunks and dispatching across rayon's global pool.
/// JITed code is pure native and holds no GIL, so we get true parallelism
/// without the sub-interpreter pool's per-call overhead.
///
/// Pointers are passed across thread boundaries as `usize`s — Rust's
/// auto-trait inference rejects raw pointers in Send closures, but
/// integers carry through fine and we re-cast inside each worker. Per
/// chunk owns a disjoint sub-slice of the caller-owned buffer, so there's
/// no aliasing.
///
/// # Safety
/// Same as `invoke_kernel`: buffers must match `in_dtype`/`out_dtype`
/// and `len` must be ≤ both buffer sizes.
pub unsafe fn invoke_kernel_parallel(
    kernel_hash: u64,
    in_dtype: Dtype,
    out_dtype: Dtype,
    in_ptr: *const u8,
    out_ptr: *mut u8,
    len: usize,
) -> Result<(), String> {
    if len == 0 {
        return Ok(());
    }
    let fn_ptr = lookup(kernel_hash)
        .ok_or_else(|| format!("kernel {} not registered", kernel_hash))?;

    let num_threads = rayon::current_num_threads().max(1);
    // Below 1 element per worker, single-thread dispatch beats spawn
    // overhead. Don't scale this with chunk size: per-element compute
    // weight is unknown, and one heavy element per worker is still worth
    // it when the kernel runs in milliseconds.
    if len < num_threads {
        let f = unsafe { as_kernel_fn(fn_ptr) };
        unsafe { f(in_ptr, out_ptr, len) };
        return Ok(());
    }

    let in_size = dtype_size(in_dtype);
    let out_size = dtype_size(out_dtype);
    let chunk_size = crate::pick_chunk_size(len, num_threads);

    let in_addr = in_ptr as usize;
    let out_addr = out_ptr as usize;

    rayon::scope(|s| {
        let mut start = 0usize;
        while start < len {
            let this = (len - start).min(chunk_size);
            let chunk_in = in_addr + start * in_size;
            let chunk_out = out_addr + start * out_size;
            s.spawn(move |_| {
                let f = unsafe { as_kernel_fn(fn_ptr) };
                unsafe { f(chunk_in as *const u8, chunk_out as *mut u8, this) };
            });
            start += this;
        }
    });
    Ok(())
}

#[inline]
fn dtype_size(d: Dtype) -> usize {
    match d {
        Dtype::I64 | Dtype::F64 => 8,
    }
}
