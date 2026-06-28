# cute-DSL + CUTLASS cheatsheet (Blackwell sm100 / tcgen05, v4.5.2)

> Working reference for building tensor-core kernels in NVIDIA's CUTLASS Python DSL ("cute-DSL"), with Flash-Attention-4 as the worked example corpus. This area is new and fast-moving with little training-data coverage — treat the **§10 review notes** as the honesty layer (corrections + what to verify on B200). FA4 source: `~/Downloads/flash-attention-main 2/flash_attn/cute`. Regenerated 2026-06-28 via multi-agent research (official docs + FA4 mining).

## 1. Orientation, setup & mental model

### What cute-DSL is (one sentence)
You write **Python** that the `nvidia-cutlass-dsl` package **traces into MLIR IR and JIT-compiles to PTX/cubin via ptxas** — the Python is a *kernel emitter*, not the kernel. [VERIFIED-DOCS: "Python kernels are compiled at runtime into CUDA device code using MLIR infrastructure and NVIDIA's ptxas toolchain."](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)

### How it relates to CUTLASS C++ / CuTe

| Layer | What it is | Relationship |
|---|---|---|
| **CuTe** (C++) | The `Layout`/`Tensor`/`TiledMMA`/`TiledCopy` algebra (shape:stride hierarchies) | The *conceptual core*. cute-DSL exposes the **same** abstractions in Python. |
| **CUTLASS C++** | Header template library (CollectiveMma, schedulers, pipelines) | cute-DSL is **complementary, NOT a replacement** — [VERIFIED-DOCS: "CUTLASS DSLs are not a replacement for the CUTLASS C++ library."](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) It *aims* to match C++ perf ("some gaps may exist"). |
| **cute-DSL** (Python) | `import cutlass.cute as cute` | [VERIFIED-DOCS: "a low level programming model that is fully consistent with CuTe C++ abstractions."](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) Same atoms (`tcgen05.mma`, TMA, pipelines), Python ergonomics + JIT specialization. |

**Mental model: "you write Python that emits PTX/MLIR."** Python runs *once* at trace time. Statements that touch **static values** (Python ints/floats, `cutlass.Constexpr`) execute in the interpreter and bake into the IR; statements that touch **dynamic IR values** (`cutlass.Int32`, a `cute.Tensor`, `cute.arch.thread_idx()`) *emit IR* instead of executing. `print()` fires at trace time (compile-time logging); `cute.printf()` is the runtime device print. The same `@cute.jit` body re-specializes (and recompiles) whenever a baked-in static value changes.

### The CUTLASS C++ side (scoped IN for qr_v2)
The eval image clones CUTLASS C++ **and has nvcc** (see version reality below), so raw-CUDA `.cu` and CUTLASS C++ are *deployable* too. But this cheatsheet is **cute-DSL-first**: the DSL wraps the *same* UMMA/TMA atoms as the C++ (`SM100_MMA_F16BF16_SS`, `UMMA::Layout_K_SW128_Atom`, `cute::TMEM::Allocator`), so C++ names appear only as cross-reference when the DSL spelling is unclear. Building the engine in C++ is **not** a capability unlock over the DSL — same PTX, same TMEM 64-row wall.

### Trace / JIT execution model (the two-stage flow)
```
Python source ──trace──▶ custom MLIR IR ──JIT (ptxas)──▶ PTX/cubin ──launch──▶ B200
   (@cute.jit / @cute.kernel bodies run ONCE here)         (cached by static signature)
```
- **`@cute.jit`** = host-side function (sets up layouts, *launches* kernels). **`@cute.kernel`** = the GPU kernel. A `@cute.kernel` **cannot** be called from plain Python or from another kernel — only launched from inside a `@cute.jit`. [VERIFIED-DOCS](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
- **JIT caching is ON by default**, keyed on the static signature; `no_cache=True` forces recompile. `cute.compile(fn, *example_args, options="...")` returns a stable compiled handle and **bypasses the cache**. Env: `CUTE_DSL_DISABLE_FILE_CACHING`, `CUTE_DSL_CACHE_DIR`. [VERIFIED-DOCS](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html)

> ⚠️ **Beta / API-unstable.** The docs themselves say "in public beta and actively evolving … interfaces and features are subject to change." Treat every API name as *verify-on-wheel*; a wrong API name costs a Modal run.

### Version reality (pin these)

| Thing | Value | Source |
|---|---|---|
| DSL wheel | `nvidia-cutlass-dsl == 4.5.2` | project pin; **the qr_v2 eval image HAS it pre-installed** ([qr_v2 eval env HAS nvcc] — verified vs live kernelbot `main`) |
| CUTLASS C++ tag | `v4.5.1` cloned → `/opt/cutlass` (`$CUTLASS_PATH`) | eval image + Modal replica |
| Eval base image | `nvidia/cuda:12.9.1-devel` (**has nvcc**) | live kernelbot `main` |
| Target arch | B200 = **sm_100a** / sm_100f (the `a`/`f` matters) | — |
| CUDA on eval | **12.9** → `cutlass.cute.experimental` is **NOT importable** (needs CUDA 13.1) | [cute-dsl Modal loop] |

★ **The "eval has no nvcc" belief was a Modal-harness artifact** (our old `modal_microbench` image = torch+triton ≠ the eval image). Raw-CUDA / CUTLASS C++ / cute-dsl are ALL deployable on the real board.

### Where the official docs live (and why they're thin)
- Overview / intro / control-flow / types / caching / AOT prose: `https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/*`
- API reference (auto-generated, **several pages truncate or 404** — e.g. `cute_math.html` 404'd): `.../pythonDSL/cute_dsl_api/{cute,cute_arch,pipeline,utils_sm100,cute_nvgpu_tcgen05,cute_nvgpu_cpasync}.html`
- C++ / Blackwell semantics (authoritative for UMMA/pipeline behavior): `https://docs.nvidia.com/cutlass/latest/media/docs/cpp/{pipeline,blackwell_functionality}.html`
- **The only public verbatim code = `examples/python/CuTeDSL/` in `github.com/NVIDIA/cutlass` (`main`).** Best: `cute/blackwell/kernel/dense_gemm/dense_gemm.py` (canonical end-to-end sm100), `cute/blackwell/kernel/{rmsnorm,reduce}/`, `dsl_tutorials/{inline_ptx,export_to_c,call_bypass_dlpack}.py`. **61 examples also ship on the image at `/opt/cutlass/examples/python/CuTeDSL/`.**
- **Authoritative ground truth = the installed wheel.** Confirm any uncertain API via `inspect.signature` / `help()` / `dir()` on `cutlass`, `cutlass.cute`, `cutlass.cute.arch`, `cutlass.cute.math`, `cutlass.pipeline`, `cutlass.utils` (verify on B200).

### How WE test it (NO gpumode burn)
The Modal test images (`modal_lab.py` etc.) are torch+triton only — **they cannot run cute-DSL**. There is a separate faithful eval-replica image: `modal_cute_lab.py::run_candidate`. Iterate there freely; only a real gpumode submission confirms the qr_v2 board.

**The faithful eval-replica image recipe** (each line cost a debug cycle):
```python
img = (modal.Image
    .from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git","curl","gcc-13","g++-13","clang-18")
    .pip_install("numpy~=2.3")
    .pip_install("torch==2.12.0")                 # ⚠️ NOT 2.11 — see below
    .pip_install("nvidia-cutlass-dsl==4.5.2","cuda-core[cu13]","cuda-python==13.0")
    .run_commands("git clone --branch v4.5.1 ... /opt/cutlass")
    .env({"CUTLASS_PATH":"/opt/cutlass"}))
# → nvcc + torch 2.12.0+cu130 + cutlass 4.5.2 + cuda.bindings 13.0.3 + triton 3.7.0
```
> ⚠️ **The old `debian_slim + torch-cu128 (2.11)` image SIGABRTs** (`LLVM ERROR: unsupported operation`) — a cuda-python-13 vs torch-2.11-cuda-bindings(<13) ABI clash. **`torch==2.12.0` fixes it.**

**Profiling reality:** `ncu`/`nsys` are **gVisor-dead on Modal** (no perf counters). Your only tools are **timing-ablation** and **SASS/resource dumping**: `fn.dump_to_object(path)` then `cuobjdump -res-usage` to read regs/spills (the FA4 `setmaxregister` workflow demands `n_spills == 0`).

### Minimal skeleton (the SIGSEGV-free launch shape)
```python
import cutlass, cutlass.cute as cute
from cutlass import Constexpr
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def my_kernel(gA: cute.Tensor, gC: cute.Tensor, n: cutlass.Int32, FLAG: Constexpr[bool]):
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(FLAG):              # compile-time specialization (untaken path not emitted)
        ...
    for i in cutlass.range_constexpr(4):      # fully unrolled at trace time
        ...
    cute.arch.barrier()

@cute.jit
def host(mA: cute.Tensor, mC: cute.Tensor, n: cutlass.Int32, FLAG: Constexpr[bool] = True):
    my_kernel(mA, mC, n, FLAG).launch(grid=[gx,1,1], block=[256,1,1])   # 3-elt seqs; cluster= only if atoms need it

a = from_dlpack(torch_a).mark_layout_dynamic()         # the ONLY torch→cute bridge
compiled = cute.compile(host, a, c, n_val, True, options="--generate-line-info")
compiled(a, c, n_val, True)                            # subsequent calls skip JIT
```
**Top launch-failure causes (verify-on-B200):** grid under-provisioned vs the per-CTA slice index; a slice coord that leaves a leftover singleton mode the MMA/TMA atom didn't expect; `cluster=` not matching what the tiled_mma/TMA atoms were built for. Access tensors via TV-layout tiled-copy + `make_fragment_like`/`cute.copy`, **not** scalar `t[i]` (scalar `Hm[r,c]` *does* work in dynamic loops but is slow — correctness floors only).

> ⛔ **Hard constraint, every section:** a submission may not contain the substrings **`stream`** or **`graph`** anywhere (naive static scan — docstrings count; `cute.arch.sm_id()`'s docstring literally says "Streaming Multiprocessor"). `grep -niE "stream|graph"` must be empty before every submit. `cute.compile`/`.launch` work with **no** stream arg — never copy the `.launch(..., stream=...)` plumbing from the docs examples.

---

All confirmed. `const_expr`/`range_constexpr` are exported from `cutlass` (defined in `base_dsl/ast_helpers.py`). `bitcast` exists as a module-level helper but the project's empirically-validated form is the value `.bitcast(dtype)` method. `from_dlpack` at runtime.py:390, `mark_layout_dynamic(leading_dim=None)`. I have enough verified ground truth. Writing the section.

## 2. Core language & runtime

> Verified against the cached CuTeDSL source tree (`.../cutlass/python/CuTeDSL/`, the wheel shipped with flash-attn 2.8.1) where a file:line is cited as `[SRC]`; FA4 examples cited as `file:line`; docs cited by URL. Items not directly read are marked **(verify on B200)**. Pinned to `nvidia-cutlass-dsl==4.5.2`, sm_100a, CUDA 12.9.

### 2.1 The two decorators

| | `@cute.jit` | `@cute.kernel` |
|---|---|---|
| Runs on | **host** (sets up layouts, launches kernels) | **device** (the GPU kernel body) |
| Callable from | Python, other `@jit`, `@kernel` | **only** from inside a `@jit` (never from Python, never from another `@kernel`) |
| Returns | a value / nothing | a *launchable handle* — you call `.launch(...)` on it |
| Class-method form | `@cute.jit def __call__(self,...)` works | `@cute.kernel def kernel(self,...)` works; `@cute.kernel()` with parens also valid |

Calling matrix (docs [introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)): `jit→jit` ✅ inlined · `jit→kernel` ✅ (a launch) · `kernel→jit` ✅ inlined · `Python→kernel` ❌ · `kernel→kernel` ❌.

The keystone idiom (fixes the launch SIGSEGV — a `@kernel` must be launched, never called):
```python
@cute.kernel
def my_kernel(gA: cute.Tensor, gC: cute.Tensor, n: cutlass.Int32, FLAG: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    ...

@cute.jit
def host(mA: cute.Tensor, mC: cute.Tensor, n: cutlass.Int32, FLAG: cutlass.Constexpr = True):
    my_kernel(mA, mC, n, FLAG).launch(grid=[gx,1,1], block=[256,1,1])   # optional cluster=[1,c,1]
```
`@cute.kernel` also accepts launch config as decorator args (`grid=/block=/cluster=/smem=/use_pdl=/cooperative=`), but in FA4 and the validated qr engine the config is passed at the **`.launch()` call site** — prefer that form.

### 2.2 `.launch()` — grid/block/cluster

- `grid`, `block`, `cluster` are **3-element sequences** (`[x,y,z]`). A wrong `block` length OR a `grid` that under-provisions vs the per-CTA slice index = the classic launch SIGSEGV; debug with `cute.printf` + `cute.pretty_str` at trace time. **(verify on B200)**
- `cluster=` must match the cluster the `tiled_mma`/TMA atoms were built for; **omit for 1-CTA**; a mismatch is an illegal launch. FA4: `cluster=[1, cfg.cluster_n, 1] if cutlass.const_expr(cfg.cluster_n > 1) else None`.
- **No stream argument** in the qr submission path (the `stream` substring is banned — see the global constraint). `.launch(..., stream=...)` exists (export example uses `stream: cuda.CUstream`) but do **not** copy it in.
- Sizes come from layout introspection, e.g. `grid=[cute.size(gC, mode=[1]),1,1]`, `block=[cute.size(tv_layout, mode=[0]),1,1]` (`elementwise_add.py`).

### 2.3 `cute.compile` / JIT caching

- `compiled = cute.compile(host_fn, *example_args, options="--generate-line-info")` → a stable `JitCompiledFunction`; call `compiled(*same_args)` thereafter (subsequent calls skip JIT). `cute.compile` **bypasses the cache** and always compiles, returning a fixed executor handle ([JIT caching docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html)).
- **What gets baked vs passed at call time:** every `cutlass.Constexpr`-typed param and any non-`cute.Tensor` python value (shapes, tile sizes, dtypes, layouts) is **frozen into the IR at trace time**. `cute.Tensor` args are passed again at the call with the **same positional list**; only pointer + dimensions marked `.mark_layout_dynamic()` / `.mark_compact_shape_dynamic()` vary across calls. The compiled signature is keyed on the static parts.
- Direct `@cute.jit` calls (no `cute.compile`) **are cached by default** (in-memory + on-disk JIT-executor map). Knobs: `no_cache=True` forces recompile; env `CUTE_DSL_DISABLE_FILE_CACHING=True`, `CUTE_DSL_CACHE_DIR=<path>`; default max 1000 cache files.
- `--generate-line-info` is accepted (cuda-gdb / line-mapped SASS).

### 2.4 cutlass scalar types: static vs dynamic — the central model

| Form | Nature | Use |
|---|---|---|
| bare python `int/float/bool`, or `cutlass.Constexpr` / `Constexpr[T]` param | **compile-time (static)** — baked into IR, drives specialization | tile sizes, NB, dtype flags, loop bounds you want unrolled |
| `cutlass.Int32` / `Float32` / `Boolean` param, or anything a device op returns (`cute.arch.thread_idx()`) | **dynamic IR (SSA) value** — a runtime register; arithmetic **traces** into IR | matrix dim `n`, indices, per-thread data |

- Types: `cutlass.{Int8,Int16,Int32,Int64,Uint*,Float16,Float32,Float64,Boolean,TFloat32,BFloat16,Float8*}`; all derive `cutlass.Numeric` with `.width`, `.is_integer`, `.is_float`. `Constexpr` is a real class `[SRC base_dsl/typing.py:1715]`. **(low-precision dtype spellings: verify on B200)**
- `const_expr` and `range_constexpr` are exported from the top-level `cutlass` namespace `[SRC cutlass/__init__.py:32,34]` (defined in `base_dsl/ast_helpers.py`).
- Casts: `value.to(cutlass.Float32)`, `frag.load().to(Float32)`; construct via `cutlass.Int32(x)`, `cutlass.Boolean(pred)`.
- **`cute.assume(x, divby=8)`** injects a divisibility guarantee on a dynamic int (unlocks vectorization / layout division); FA4 `cute.assume(s, divby=128//width)` on all-but-last strides.

### 2.5 Control-flow tracing — preprocessor, `const_expr`, `range`, pitfalls

The DSL parses the Python AST per statement and decides compile-time vs IR-emit.

| Construct | Behavior |
|---|---|
| `cutlass.range(n)` / bare `range(n)` | **always emits an IR loop** (even with python bounds). `range(n, unroll=N)`, `range(n, prefetch_stages=N)` (SW-pipelined) |
| `cutlass.range_constexpr(n)` | runs in the **Python interpreter, fully unrolled**; bounds **must be Constexpr** |
| `if pred:` (dynamic predicate) | emits an **IR branch** |
| `if cutlass.const_expr(flag):` | **compile-time specialization** — untaken path not emitted (the kernel-specialization idiom) |
| `while`(dynamic cond) | IR loop; `const_expr` cond → compile-time |

**Pitfalls** ([control-flow docs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html)):
- ❌ Never feed a **dynamic IR value** to native python flow or a `*_constexpr` (`for i in range_constexpr(dynamic_n)` fails).
- Dynamic (IR) control flow does **not** support `break`/`continue`, raising exceptions, values escaping block scope, or **changing a variable's type** inside the block.
- **Closures are not supported in dynamic control flow** (project-validated, M3c): nested `def`s capturing pipeline objects fail — **INLINE** the mma/readback bodies.
- `print(...)` inside a kernel runs at **trace/JIT time**; for runtime device print use `cute.printf(...)` / `cute.print_tensor(...)`. (M6 heisenbug note: a `cute.printf` can mask a timing race — its absence ≠ correctness.)

**Helpers** (`from cutlass.cutlass_dsl import dsl_user_op, if_generate, T`):
- `@dsl_user_op` — wrap a function as an IR-emitting op (the template for inline-PTX intrinsics; takes hidden `*, loc=None, ip=None`). `if_generate` confirmed present `[SRC cutlass_dsl]`.
- `if_generate(cond, then_fn[, else_fn])` — the **functional `if`** that emits a branch region when a python `if` can't be used (e.g. FA4 gates `extra_tx_count==0` vs the expect-tx path: `pipeline.py:304-327`). With a `const_expr` cond it folds away.
- `T` = MLIR type builder (`T.i32()`, `T.bool()`) for raw-op / inline-asm emission.

### 2.6 `from_dlpack` / torch interop

`from cutlass.cute.runtime import from_dlpack` `[SRC runtime.py:390]` — the **only** torch→cute bridge:
```python
a = from_dlpack(torch_a, assumed_align=16).mark_layout_dynamic()   # [SRC runtime.py:187]
```
- `assumed_align` must be a **real** alignment of the torch buffer or TMA faults (align `[B,n,n]` to ≥16B).
- `.mark_layout_dynamic(leading_dim=None)` makes shape/strides runtime-dynamic so one compile serves many sizes; omit (or pass concrete tensors) to bake the layout static. `.mark_compact_shape_dynamic(...)` `[SRC runtime.py:210]` for the compact-stride case.
- **DLPack mishandles shape-1 modes** (collapses stride→1, breaks alignment) → use the pointer-bypass path for those.
- Access tensors via TV-layout tiled copies + `make_fragment_like`/`cute.copy`, **not** scalar `t[i]` for the perf path. Caveat/reconcile: scalar indexing like `Hm[r,c]` **does** work inside a `@cute.jit` dynamic loop (project-validated) — it's correct but slow, fine for correctness-floor code, not the hot path.
- torch helpers: `import cutlass.torch as cutlass_torch` → `cutlass_torch.dtype(cutlass_dtype)`, `cutlass_torch.current_stream()`. Pointer bypass: `from cutlass.cute.runtime import make_ptr` + `cute.make_tensor(ptr, cute.make_ordered_layout(...))`.

### 2.7 `cute.arch` intrinsics

`cute.arch` = thin NVVM wrappers `[SRC cutlass/cute/arch/]`. All idx/dim return `Int32` (or a 3-tuple):

| Call | Returns / note |
|---|---|
| `thread_idx()`, `block_idx()`, `block_dim()`, `grid_dim()` | 3-tuple of `Int32` |
| `warp_idx()`, `lane_idx()` | `Int32` |
| `make_warp_uniform(x)` | **REQUIRED** before branching on `warp_idx` (uniformizes); FA4 `warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())` |
| `cluster_dim()/cluster_idx()/block_idx_in_cluster()` | cluster geometry |
| `sync_threads()` | `__syncthreads`; gives cross-warp gmem visibility |
| `barrier(barrier_id=None, number_of_threads=None)` | named barrier arrive+wait; `barrier_arrive(...)` = arrive-only |
| `sync_warp(mask=0xffffffff)` | warp sync |
| `elect_one()` | **context manager**, returns `IfOpRegion` `[SRC arch/elect.py:61]` — `with cute.arch.elect_one():` |
| `vote_ballot_sync/any/all/uni_sync(pred, mask)` | warp vote |
| `warp_reduction_max/_sum(...)`, `shuffle_sync_bfly(val, offset)` | warp reduce / butterfly (does **not** cross warp boundaries) |
| `mbarrier_init/_init_fence/_arrive/_wait/_try_wait`, `cp_async_commit_group/_wait_group` | async sync (see pipeline section) |

**`elect_one` trap:** do **NOT** wrap `cute.copy(TMA...)` or `cute.gemm(...)` in `elect_one` — they self-elect / are issued warp-uniform; wrapping deadlocks. (Conversely `tcgen05.commit` **must** be elect-one'd — see the tcgen05 section.) Named barriers: IDs 0–15 (id 0 reserved for `sync_threads`); FA4 enum starts at 1 (`named_barrier.py:6`).

### 2.8 `cute.math` (NOT `cute.arch.sqrt`) + bitcast

`cute.math` operates on tensor/SSA fragments `[SRC cutlass/cute/math.py]`:

| Op | Signature |
|---|---|
| `cute.math.sqrt(a, fastmath=False)` | `[SRC math.py:277]` — **note default `False`**; FA4/qr pass `fastmath=True` |
| `cute.math.rsqrt(a, fastmath=False)` | FA4 rmsnorm |
| `cute.math.exp2(a, fastmath=False)` | `[SRC math.py:149]` — base-2 (HW MUFU); softmax uses `exp2`, not `exp` |
| `cute.math.log2(a, ...)`, `cute.math.exp(...)` | present |

- ⚠️ **`cute.arch.sqrt` does NOT exist** — the M1 build-block fix was `cute.arch.sqrt → cute.math.sqrt(x, fastmath=True)`.
- Per-scalar NVVM intrinsics live on `cute.arch`: `exp2(a)->Float32`, `fmax(a,b)`, `rcp_approx(a)`, `fma_packed_f32x2(...)`, `add_packed_f32x2(...)`.

**Bitcast — use the value `.bitcast(dtype)` METHOD, not a module function** (project-validated, each cost a Modal run):
```python
xi = x.bitcast(cutlass.Int32)
hi = (xi & cutlass.Int32(-8192)).bitcast(cutlass.Float32)   # -8192 = 0xFFFFE000, clears low 13 mantissa bits
lo = x - hi                                                  # tf32x3 hi/lo split, validated EXACT
```
Three **dead alternatives** that silently collapse to 1×tf32 under the optimizer (rel stuck ~8.8e-4):
- `cute.bitcast(...)` as a free function — the in-tree helper `[SRC base_dsl/_mlir_helpers/arith.py:114 bitcast(src, res_elem_type)]` is low-level; the **value method `.bitcast()` is the validated path** (the build-guide's `cute.bitcast` reference is wrong).
- `cutlass.and_` / `cute.core.and_` — **logical** and → produces nan. Use **python `&`** (bitwise) on the `Int32` view.
- The arithmetic Dekker split `c-(c-x)` AND the `x.to(TFloat32).to(Float32)` round-trip both get optimized away — **only the bit-AND survives.**
- Tensor-level reinterpret (different purpose): `cute.recast_tensor(tensor, NewType)`.

### 2.9 Module / import map (4.5.2)

```python
import cutlass                         # Int32, Float32, Boolean, Constexpr, const_expr, range, range_constexpr
import cutlass.cute as cute            # jit, kernel, compile, gemm, copy, make_tensor, printf, pretty_str, recast_tensor, assume
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import dsl_user_op, if_generate, T
import cutlass.torch as cutlass_torch
# cute.arch  (idx/dims, barriers, elect_one, mbarrier_*, fences)
# cute.math  (sqrt/rsqrt/exp2/log2 — fastmath kwarg)
# cute.nvgpu.{cpasync, tcgen05, warp}   ;  cutlass.pipeline  ;  cutlass.utils(.blackwell_helpers)
```
**4.5.2 deltas to remember:** no `warp_specialize` (Gluon-ism → manual `warp_idx` branch + `setmaxregister`); `cutlass.cute.experimental` raises `NotImplementedError` below CUDA 13.1 (eval = 12.9 → unavailable). All `[SRC]` items above were read from the cached wheel tree; everything tagged **(verify on B200)** should be confirmed via `inspect.signature` / `dir()` against the installed `nvidia-cutlass-dsl==4.5.2` and a real run.

---

I have enough context on the existing style. Now I'll write the section.

## 3. Layouts, tensors, partitioning & memory allocation

> Provenance tags reuse the existing cheatsheet legend (`EW`/`DG`/`G1`/`T1`/`LIB:`/`GG`). `FA4:` = Flash-Attention-4 source under `flash_attn/cute/` (file:line). `[VERIFIED-SRC]` = verbatim from official CUTLASS `examples/python/CuTeDSL/` or the installed wheel; `[VERIFIED-DOCS]` = docs.nvidia.com/cutlass; `(verify on B200)` = reasoned, not confirmed against the 4.5.2 wheel. **A wrong API name is worse than an omission — anything below the confidence line is flagged.**

### 3.1 The CuTe layout model — `(shape):(stride)`

A **Layout** is a function from a logical coordinate → a linear offset, written `shape:stride`. This is the #1 source of "rank didn't reduce → launch SIGSEGV" bugs, so internalize the algebra before the API.

- **Shape** = the logical extents; **stride** = the multiplier per mode. `offset(c) = dot(coord, stride)`. Example `(8,4):(1,8)` = an 8×4 column-major tile (mode-0 contiguous).
- **Hierarchical / nested layouts**: a mode can itself be a layout. `((R,8),1,4,1):((32,1),0,8,0)` (the validated swizzled MMA-operand layout from memory) means mode-0 is a *pair* `(R,8)` with strides `(32,1)`, and stride-`0` modes are **broadcast** (the same address for every index — how a singleton/replicated dim is encoded).
- **`cosize`** = the smallest buffer size that holds the layout = `1 + max offset` = what you must allocate. `cute.cosize(layout)` is exactly what `MemRange[dtype, cute.cosize(...)]` sizes (existing §2).
- **`size`** = product of shape (logical element count); **`rank`** = number of top-level modes; **`depth`** = nesting levels.

```python
cute.make_layout((M, N), stride=(s0, s1))     # explicit shape:stride            [VERIFIED-SRC]
cute.make_layout((M, N))                       # default = column-major (generalized)
cute.make_ordered_layout((4, 32), order=(1,0)) # EW: pick which mode is contiguous via `order`
cute.size(t, mode=[i]) ; cute.cosize(layout) ; cute.rank(t) ; cute.depth(layout)
```

**`make_ordered_layout(shape, order)`** is the ergonomic alternative to hand-computing strides: `order` ranks modes from fastest-varying (`0`) to slowest; CuTe fills in the compact strides. `order=(1,0)` = mode-1 contiguous (row-major); `order=(0,1)` = mode-0 contiguous (col-major). FA4 uses it to vectorize along the contiguous axis: `make_ordered_layout(..., order=(1,0))` so threads stride along the contiguous row (`FA4: copy_utils.py:81-106`).

### 3.2 Layout operations — divide family (the partition algebra)

These produce the per-CTA / per-thread sub-tensors. The distinction matters because **the wrong one leaves a singleton mode the MMA/TMA atom rejects** → SIGSEGV.

| Op | Produces | Use |
|---|---|---|
| `cute.zipped_divide(t, tiler)` | `[(tile_modes...), (rest_modes...)]` — tile grouped into mode-0, tile-iteration into mode-1 | EW per-block tiling; pick a tile with `t[((None,None), idx)]` |
| `cute.tiled_divide(t, tiler)` | like zipped but tile modes stay un-grouped (flat) | build the VMNK cta layout (`G1`) |
| `cute.flat_divide(t, tiler)` | fully flattened tile + rest | epilogue subtiling (`G3`) |
| `cute.logical_divide(t, tiler)` | nested `(tile, rest)` per mode (the primitive the others wrap) | manual control |
| `cute.local_tile(t, tiler, coord, proj=...)` | one tile at `coord`; `proj` projects out kept modes | per-CTA matrix slice (THE QR pattern) |
| `cute.local_partition(t, layout, idx)` | per-**thread** slice of a tile by a thread-layout | thread-level partition (non-TV path) |

```python
# local_tile: slice mA[B,n,n] -> a [n,n] view for this CTA (None keeps a whole mode)   [VERIFIED-SRC, G1/DG]
gMat = cute.local_tile(mA_bnn, tiler=(1, n, n), coord=(bidx, None, None))
gA   = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))  # GEMM idiom

# local_partition: distribute a tile across `thr_layout` threads, take this thread's slice
tThr = cute.local_partition(blkTile, thr_layout, tidx)                            (verify on B200)
```

Other layout helpers (existing §2, retained): `cute.group_modes(t, lo, hi)` (collapse modes `[lo,hi)` into one — required before `tma_partition`), `cute.select(layout, mode=[...])` (pick a sub-layout), `cute.coalesce` / `cute.composition` / `cute.complement` (the lower-level algebra; coalesce merges adjacent compatible modes, composition `A∘B` reindexes A by B). `cute.make_identity_tensor(shape)` builds a coordinate tensor for predication.

**Slicing semantics** (load-bearing): `t[(None, i)]` keeps mode-0 and **fixes** mode-1 = `i`; `None == ":"` (keep the whole mode). A coord that fixes *every* mode of a tiler leaves a rank-0 result; one that leaves a stray singleton confuses the atom. Debug both at trace time: `print(cute.pretty_str(gMat))` + `cute.printf("bidx={} rank={}", bidx, cute.rank(gMat))`.

### 3.3 Tensors — `make_tensor`, `make_fragment`, `from_dlpack`

```python
# host bridge (the ONLY torch->cute path)                                          [VERIFIED-SRC]
from cutlass.cute.runtime import from_dlpack
a = from_dlpack(torch_a, assumed_align=16).mark_layout_dynamic()       # dynamic layout
# .mark_compact_shape_dynamic(mode=1, divisibility=k)  for a specific dynamic dim

# build a tensor from a pointer + layout (in-kernel, or pointer-bypass host path)
t  = cute.make_tensor(ptr, layout)                                     # ptr: cute.Pointer  [VERIFIED-SRC]
t  = cute.make_tensor(ptr, cute.make_ordered_layout((m,k,l), order=(0,1,2)))

# fragments (RMEM register tensors)
frag = cute.make_fragment(layout, dtype)        # explicit RMEM fragment
frag = cute.make_fragment_like(part, dtype)     # RMEM matching a partition's shape  [VERIFIED-SRC]
rt   = cute.make_rmem_tensor(shape, dtype)      # explicit RMEM tensor
```

- **`assumed_align` must be a real alignment** of the torch buffer or TMA faults; align `[B,n,n]` ≥16B. **DLPack mishandles shape-1 modes** (collapses stride→1, breaking alignment) → use the `make_ptr` pointer-bypass path for those (`[VERIFIED-SRC]` call_bypass_dlpack.py).
- **`make_fragment_like`** copies a partition's TV shape into registers — the standard way to size the RMEM destination of a `cute.copy`. FA4: `cute.make_rmem_tensor(thr_copy.partition_D(tScS).shape, acc_dtype)` (`FA4: sm100_hd256_2cta_fmha_forward.py:1551`).
- **Convert-on-copy** (dtype cast fused into a copy): `frag.store(src.load().to(dst.element_type))` then copy (`FA4: copy_utils.py:16-32`). Panel reductions must stay fp32.
- **Scalar indexing**: `Hm[r,c]` *is* allowed inside `@cute.jit` dynamic loops (validated, memory), but it is slow — for perf use the partitioned/TV path + `cute.copy`. Scalar indexing is for correctness-floor code only, not the hot loop.

### 3.4 TV (thread-value) layouts & tiled partitioning

A **TV layout** maps `(thread_id, value_id) → logical coordinate` — it is how a tiled copy/MMA decides which thread owns which elements.

```python
# EW: build a thread x value TV layout, then partition source/dest
tv_layout = cute.make_layout_tv(thr_layout, val_layout)     # -> (tiler_mn, tv_layout)   [VERIFIED-SRC]
thr_copy  = tiled_copy.get_slice(tidx)         # this thread's view
tSrc = thr_copy.partition_S(gSrc)              # partition the SOURCE  (gmem/smem)
tDst = thr_copy.partition_D(gDst)              # partition the DEST    (rmem/smem)
cute.copy(tiled_copy, tSrc, tDst)
```

- `partition_S` / `partition_D` consume the TV layout from a `tiled_copy`/`tiled_mma` `.get_slice(tidx)`. The MMA analog: `tiled_mma.make_fragment_A(sA)` / `make_fragment_B(sB)` / `make_fragment_C(acc_shape)` and `tiled_mma.partition_shape_C(tiler_mn)` (existing §4).
- For non-vectorized/manual cases, `cute.local_partition(tile, thr_layout, tidx)` distributes by a plain thread-layout (no value sub-layout).

### 3.5 SMEM layouts & swizzle (NVMMASharedLayout, 128B swizzle)

For tcgen05, A/B SMEM operands must satisfy **both** TMA and UMMA layout rules simultaneously. Don't hand-build them — use the helpers.

```python
from cutlass.utils import blackwell_helpers as sm100_utils
import cutlass.cute.nvgpu.tcgen05 as tcgen05

# convenience: full A/B/epi smem layout for a tiled_mma (handles swizzle + staging)  [VERIFIED-SRC, DG]
sA_layout  = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, a_dtype, num_stages)
sB_layout  = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, b_dtype, num_stages)
sEpi       = sm100_utils.make_smem_layout_epi(...)

# lower-level: a single swizzle atom by kind
atom = tcgen05.make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, element_type)        [VERIFIED-SRC]
```

**`SmemLayoutAtomKind`** (4.5.2 enum) — the swizzle/atomicity choices:

| Kind | Meaning |
|---|---|
| `K_INTER`, `K_SW32`, `K_SW64`, `K_SW128` | K-major, {interleave, 32B, 64B, **128B**} swizzle |
| `MN_INTER`, `MN_SW32`, `MN_SW64`, `MN_SW128` | MN-major variants |
| `MN_SW128_32B` | 128B swizzle, **32B atomicity** — required for transposed (MN-major) operands per the MMA docstrings |

- **`SW128` = 128-byte swizzle** = the common GEMM choice; it permutes within a 128B segment so smem-bank conflicts on the wide MMA reads cancel. The C++/Gluon name for this swizzled layout is **NVMMASharedLayout** (`UMMA::Layout_K_SW128_Atom<T>` in C++); the DSL spells it `SmemLayoutAtomKind.*_SW128` (`[VERIFIED-SRC]` atom-kind; NVMMASharedLayout name `[VERIFIED-DOCS]`).
- **`Swizzle<B,M,S>`** is the underlying functor (B = bits, M = base, S = shift); 128B swizzle of fp32 ≈ `Swizzle<3,4,3>`. You rarely construct it directly — `make_smem_layout_atom` picks it. (verify exact `<B,M,S>` on B200.)
- A swizzled SMEM layout in the DSL is a **`ComposedLayout`** = plain layout ∘ swizzle. Allocate it by splitting `.outer` (plain) and `.inner` (swizzle) (existing §2, retained):

```python
sA = smem.allocate_tensor(element_type=io_dtype, layout=sA_layout.outer,
                          byte_alignment=128, swizzle=sA_layout.inner)              [VERIFIED-SRC]
# alt: storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
```

**Hand-filling a swizzled operand works** (refutes "swizzled coord-write silently corrupts"): writing by *logical* coord `sX[(row, ki), 0, kb, 0] = val` (k = kb*8 + ki) into a `((R,8),1,4,1):((32,1),0,8,0)` layout is validated on B200 — you may apply a unit-lower mask during the fill. This is the only path when feeding a transposed/masked operand that TMA can't produce (memory; see §3.7 and the pipeline section on `PipelineAsyncUmma`).

### 3.6 SMEM allocation — `SmemAllocator`, `@cute.struct`

```python
smem    = cutlass.utils.SmemAllocator()                  # NOTE: cutlass.utils, not cutlass.cute  [VERIFIED-SRC]
storage = smem.allocate(SharedStorage)                   # SharedStorage = @cute.struct
sA      = smem.allocate_tensor(element_type=dt, layout=L.outer, byte_alignment=128, swizzle=L.inner)
```

```python
@cute.struct
class SharedStorage:
    ab_mbar_ptr:  cute.struct.MemRange[cutlass.Int64, ab_stages * 2]   # mbarrier array (2/stage)
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf:  cutlass.Int32                                   # holds the TMEM base addr
    sA: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sA_layout)], 128]
    sB: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sB_layout)], 128]
# access: storage.ab_mbar_ptr.data_ptr()  /  storage.tmem_holding_buf.ptr  /  storage.sA.get_tensor(...)
```

- `MemRange[T, N]` reserves N elements of T; size SMEM tensors with `cute.cosize(layout)`. `Align[..., 128]` enforces the 128B alignment swizzled MMA operands and mbarrier arrays require.
- **SMEM budget = 228 KB/SM on B200** (the hard occupancy gate alongside the 64K-reg/SM pool). Size your ring as `num_stages × cosize(sA_layout) × bytes` + mbarrier bytes + accumulator-staging smem ≤ 228 KB; overflow = launch failure (verify exact KB on B200; documented 227-228 KB).
- **Pitfall (memory #5/#9)**: a *raw-SMEM* `partition_D` for staging a transposed intermediate (W2/dV) can fail the layout division → either fix the layout or stage through gmem scratch. And one `sStage[128,256]` cannot hold **both** W2 and dV alive at once — fill W2 into its own buffer *before* the output-tile loop. (Both validated the hard way.)

### 3.7 TMEM allocation — `TmemAllocator`, the 512-column wall

TMEM is a **single per-SM resource: 128 lanes × 512 columns of 32-bit = 256 KB/SM**. The accumulator lives here, not in registers — this is the sm100-specific occupancy gate.

```python
from cutlass import utils
tmem = utils.TmemAllocator(arch="sm_100", is_two_cta=use_2cta,
                           two_cta_tmem_dealloc_mbar_ptr=mbar)   # mbar REQUIRED iff is_two_cta  [VERIFIED-SRC]
# allocator-warp only:
tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))      # = 512 cols
tmem.wait_for_alloc()                       # NamedBarrier so ALL warps see the base ptr
ptr = tmem.retrieve_ptr(cutlass.Float32)    # MUST come after wait_for_alloc in every reader warp
... cute.gemm(tiled_mma, tAcc, ...) ...     # accumulate into TMEM
tmem.relinquish_alloc_permit()              # promise no more allocations
tmem.free(ptr)                              # dealloc
# low-level alt: cute.arch.{alloc_tmem, retrieve_tmem_ptr, relinquish_tmem_alloc_permit, dealloc_tmem}
```

**Column budget — the occupancy arithmetic (`[VERIFIED-SRC]` budget + the project's falsified per-matrix engine):**

| Allocation | Cols consumed | CTAs/SM |
|---|---|---|
| 512-col F32 acc (e.g. `[128,512]`) | 512 = **all of TMEM** | **1 CTA/SM** |
| 256-col F32 acc (e.g. `[128,256]`) | 256 | 2 CTAs/SM |
| 128-col | 128 | up to 4 (if regs/smem also fit) |

- **Allocation rule** (validated): `num_columns % 32 == 0` **AND** a power of two, in `[32, 512]` (`get_min/max_tmem_alloc_cols`). `[128,256]` acc = 256 cols.
- **The 512-col wall = your 1-CTA/SM result.** A full-width accumulator monopolizes TMEM → no co-residency → the per-matrix tcgen05 engine is bounded ≈1.0-1.2× V10. To get >1 CTA/SM you must shrink the acc (fewer N columns) or share via `TmemBufferPool`. There is also a separate **NBP ≥ 64 row wall** (the `[NBP,BW]` tile pads to 64 rows minimum → the ~2.0× serial-tcgen05 floor at n≤512) — distinct from the column budget; both bite.
- **`wait_for_alloc()` MUST precede `retrieve_ptr` in every warp** that reads the acc, or you get a garbage pointer = fault. The base address is published through `tmem_holding_buf` in SMEM.

**`TensorMemoryLayout` / `two_ctas` / 2-SM path:** the 2-SM ("two_ctas") accumulator is selected by **`CtaGroup.TWO`** on the MMA op + **`is_two_cta=True`** on the allocator (which inits the cross-CTA dealloc mbarrier; assert you passed `two_cta_tmem_dealloc_mbar_ptr`) + the `SM100_MMA_*_2x1SM_SS` atom. In C++ this is `TMEM::Allocator2Sm` vs `Allocator1Sm`; in the Python DSL it is the single `is_two_cta` flag — you do not name `TensorMemoryLayout(two_ctas)` directly for the standard path (`[VERIFIED-SRC]` flag; the standalone `TensorMemoryLayout(two_ctas)` entry point is *(verify on B200)*).

**FA4 reference (the canonical TMEM lifecycle):**
```python
tmem = cutlass.utils.TmemAllocator(storage.tmem_holding_buf.ptr,
          barrier_for_retrieve=tmem_alloc_barrier,
          allocator_warp_id=self.mma_warp_id, is_two_cta=self.use_2cta_instrs)   # FA4:flash_fwd_sm100.py:885
if warp_idx == self.mma_warp_id:
    tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
    tmem.wait_for_alloc(); tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)        # FA4:...:1187
    self.mma(...)
    tmem.relinquish_alloc_permit(); tmem_alloc_barrier.arrive_and_wait(); tmem.free(tmem_ptr)
# consumer warps get the SAME ptr via the shared barrier:
#   tmem.wait_for_alloc(); tmem_ptr = tmem.retrieve_ptr(...)                       # FA4:...:1246
```
(Note FA4 passes the holding-buf ptr + a retrieve barrier + allocator-warp-id positionally; the `arch=`/`is_two_cta=` kwarg form above is the `dense_gemm.py` style — both exist in 4.5.2; *verify which positional vs kwarg form your wheel uses*.)

### 3.8 Reading/writing the TMEM accumulator (t2r / r2t)

TMEM ↔ RMEM uses dedicated `tcgen05.ld`/`.st` copy atoms; pick the atom with the helper, build a tiled copy, then `cute.copy`, then **fence**.

```python
# pick + build the TMEM->RMEM copy                                                  [VERIFIED-SRC, DG]
copy_atom_t2r  = sm100_utils.get_tmem_load_op(cta_tile, layout_d, dt_d, dt_acc, epi_tile, use_2cta)
tiled_t2r      = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None,None,0,0)])
thr_t2r        = tiled_t2r.get_slice(tidx)
rAcc           = cute.make_rmem_tensor(thr_t2r.partition_D(tCoord).shape, dt_acc)
cute.copy(tiled_t2r, thr_t2r.partition_S(tAcc_epi), rAcc)      # TMEM -> registers
cute.arch.fence_view_async_tmem_load()                         # MANDATORY before reusing rAcc / releasing
# write-back r2t (e.g. P back into TMEM):
store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)), dt_acc)
cute.copy(tcgen05.make_tmem_copy(store_atom, tT), rSrc, tT)
cute.arch.fence_view_async_tmem_store()                        # MANDATORY before signalling the MMA warp
```

- Load atoms: `Ld32x32bOp`, `Ld16x{64,128,256}bOp`, `Ld16x32bx2Op`; store atoms `St32x32bOp`, `St16x*`; each takes a `Repetition(n)` and a `Pack`. `get_tmem_load_op` auto-selects the performant atom. **Always read the acc back in FP32** — reading the tf32 acc directly as output rounds to ~19 bits.
- **The async-fence family** (all mandatory, none in the old §3/§4): `fence_view_async_tmem_store()` after a TMEM write before signalling a pipeline; `fence_view_async_tmem_load()` after a TMEM→reg read before consuming/releasing; `fence_view_async_shared()` / `fence_proxy("async.shared", space="cta")` after an `st.shared` fill before the tensor-core proxy reads it (FA4: `flash_fwd_sm100.py:2353`, `:1557`, `flash_bwd_mla_sm100.py:1886`).

### 3.9 Confidence line — verify on B200 before relying

- `cute.local_partition(tile, thr_layout, tidx)` exact arg order — *(verify on B200)*; the TV `partition_S/_D` path is confirmed.
- `Swizzle<B,M,S>` exact `<B,M,S>` triple for fp32 128B — *(verify)*; use `make_smem_layout_atom(K_SW128, ...)` which is confirmed.
- `TmemAllocator` positional (FA4) vs `arch=`/`is_two_cta=` kwarg (dense_gemm) form — both present in 4.5.2; confirm the signature on your installed wheel with `inspect.signature(cutlass.utils.TmemAllocator)`.
- Standalone `TensorMemoryLayout(two_ctas)` — the Python entry point is the allocator `is_two_cta` flag; a directly-named `TensorMemoryLayout` is *(verify on B200)*.
- Exact SMEM/SM capacity (227 vs 228 KB) and TMEM total (256 KB) — quote `cute.arch.get_max_tmem_alloc_cols("sm_100") == 512` (confirmed) over the KB figures.
- **Do NOT** trust `cute.bitcast` (does not exist — use the `value.bitcast(dtype)` method) or `cutlass.and_` (logical→nan — use Python `&`) when manipulating layout/operand bits.

---

## 4. Copy & data movement

Module aliases: `import cutlass.cute as cute`; `from cutlass.cute.nvgpu import tcgen05, cpasync, warp`; `from cutlass.utils import blackwell_helpers as sm100_utils`. FA4 paths are relative to `flash_attn/cute/`.

### 4.0 Pick the right mover (decision table)

| Source → Dest | Arch | Mechanism | Atom / API |
|---|---|---|---|
| GMEM → RMEM (small/scalar) | all | universal copy | `cute.nvgpu.CopyUniversalOp()` |
| GMEM → SMEM, register-staged | sm80+ | `cp.async` | `cpasync.CopyG2SOp()` |
| GMEM → SMEM, bulk async | sm90+ | TMA load | `cpasync.CopyBulkTensorTileG2SOp()` |
| GMEM → SMEM, multicast (cluster) | **sm100** (2-SM) | TMA + cluster mcast | `cpasync.CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)` |
| SMEM → GMEM, bulk async | sm90+ | TMA store | `cpasync.CopyBulkTensorTileS2GOp()` |
| SMEM → GMEM, reduce-add | sm90+ | TMA reduce-store | `cpasync.CopyReduceBulkTensorTileS2GOp()` |
| SMEM → RMEM, warp-MMA feed | sm75+ | `ldmatrix` | `warp.LdMatrix8x8x16bOp` etc. |
| RMEM → SMEM | sm90 | `stmatrix` | `warp.StMatrix8x8x16bOp` etc. |
| TMEM → RMEM (read back acc) | **sm100** | `tcgen05.ld` | `tcgen05.Ld32x32bOp` / `Ld16x{64,128,256}bOp` |
| RMEM → TMEM (write acc/operand) | **sm100** | `tcgen05.st` | `tcgen05.St32x32bOp` / `St16x{64,128,256}bOp` |
| SMEM → TMEM (A-from-TMEM path) | **sm100** | s2t copy | `tcgen05.make_s2t_copy` + `get_s2t_smem_desc_tensor` |

**SM90 vs SM100 in one line:** SM90 (Hopper) wgmma reads operands from SMEM via `ldmatrix`, accumulates in **registers**. SM100 (Blackwell) `tcgen05.mma` accumulates in **TMEM**, read back via `tcgen05.ld` (`Ld32x32bOp`…); operands from SMEM-descriptor (or A-from-TMEM). The `warp.Mma*Op`/`ldmatrix` set still exists on sm100 but the throughput path is tcgen05. [VERIFIED — CUTLASS `nvgpu/{cpasync,warp,tcgen05}` `__all__`]

### 4.1 Copy atoms — `cp.async`, ld/st-matrix

cp.async group control (`cute.arch`): `cp_async_commit_group()`, `cp_async_wait_group(n)`; bulk/TMA variants `cp_async_bulk_commit_group()`, `cp_async_bulk_wait_group(group, read=None)`. [VERIFIED]

Auto-sized copy atom (clamp to 128 bits, pick cp.async vs universal by a flag) — `copy_utils.py:42`:
```python
num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
```

Convert-on-copy (fuse a dtype cast into the move) — `copy_utils.py:16`:
```python
if const_expr(src.element_type != dst.element_type):
    src_cvt = cute.make_fragment_like(src, dst.element_type)
    src_cvt.store(src.load().to(dst.element_type))
    src = src_cvt
cute.copy(atom, src, dst, pred=pred)
```
Panel reductions must stay FP32 — convert only on the store to the output buffer, never the accumulator.

### 4.2 TMA — descriptor creation

TMA itself is sm90+; the sm100 addition is the 2-SM **multicast** atom (`CtaGroup.TWO`). The atoms must be built **host/prologue side** and are returned together with a *retiled* tensor view.

For MMA operands use the sm100 helpers (they wire in the MMA tiling + cluster) — `sm100_hd256_2cta_fmha_forward.py:457`:
```python
tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
    tma_load_op, q, q_smem_layout, self.qk_mma_tiler, qk_tiled_mma, self.cluster_layout_vmnk.shape)
tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
    tma_load_op, k, k_smem_layout, self.qk_mma_tiler, qk_tiled_mma, self.cluster_layout_vmnk.shape)
```
- **Feed FP32 GMEM into a TF32 MMA via TMA** with `internal_type=cutlass.TFloat32` (in `make_tiled_tma_atom_A`). [VERIFIED — `dense_gemm.py`]
- Generic (non-MMA, e.g. the C/output store): `cpasync.make_tiled_tma_atom(op, gmem_tensor, smem_layout, cta_tiler)` — "automatically determines the bulk tensor async copy instruction." Store path uses `CopyBulkTensorTileS2GOp()`. [VERIFIED]
- Atom selector by cluster/SM count: `sm100_utils.cluster_shape_to_tma_atom_A/B/SFB(cluster_shape_mnk, tiled_mma.thr_id)` — picks the multicast op when 2-SM. [VERIFIED — `blackwell_helpers.py`]
- Per-stage byte count for the pipeline: `cute.size_in_bytes(dtype, smem_layout)` (or `cute.size_in_bytes(dtype, ...)`). [VERIFIED — FA4 `:487`]

**Prefetch the descriptor once** (warp 0) to warm the cache — `flash_fwd_sm100.py:855`:
```python
if warp_idx == 0:
    for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        if const_expr(tma_atom is not None):
            cpasync.prefetch_descriptor(tma_atom)
```
Related descriptor lifecycle helpers (`cpasync`): `update_tma_descriptor`, `copy_tensormap`, `fence_tma_desc_acquire`, `fence_tma_desc_release`, `cp_fence_tma_desc_release`; dynamic-descriptor management via `cutlass.utils.tensormap_manager`. [VERIFIED — `cpasync.__all__`]

### 4.3 TMA — partition + load/store

`cpasync.tma_partition` reshapes grouped smem/gmem tensors into the atom-friendly `((atom_v, rest_v), Rest)` form. Args: `(atom, mcast_cta_coord, cta_layout, grouped_smem, grouped_gmem)`. The `0` + `make_layout(1)` form = no multicast — `flash_fwd_sm100.py:1415`:
```python
tKsK, tKgK = cpasync.tma_partition(
    tma_atom_K, 0, cute.make_layout(1),
    cute.group_modes(sK, 0, 3),
    cute.group_modes(tSgK, 0, 3))
```

The load itself — `cute.copy` with `tma_bar_ptr=` (the mbarrier the TMA completes on) and an optional `mcast_mask=` — `dense_gemm.py` / `sm100_hd256_2cta_fmha_forward.py:947`:
```python
q_handle = load_q_producer.acquire_and_advance()        # pipeline handle
cute.copy(tma_atom_q, tQgQ[None, iter], tQsQ[None, q_handle.index],
          tma_bar_ptr=q_handle.barrier)                 # mbarrier armed by the TMA
# multicast variant:
mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, coord, mcast_mode=2)
cute.copy(tma_atom_a, gA, sA, tma_bar_ptr=bar, mcast_mask=mask)
```
Store (S→G): `cute.copy(tma_atom_c, sC_src, gC_dst)`. Reusable closures wiring partition + pipeline barrier: `copy_utils.py:324` `tma_get_copy_fn` → returns `copy(src_idx, dst_idx)`; `copy_utils.py:363` `tma_producer_copy_fn` pulls the barrier out of a `PipelineState` via `pipeline.producer_get_barrier(state)`.

### 4.4 TMA — arrive-and-expect-tx (the completion mechanism)

A TMA load does **not** arrive an mbarrier with a thread-count; it arms a **transaction barrier** by byte count. The TMA instruction decrements the expected-tx as bytes land; when expected-tx hits 0 the barrier's phase flips → consumers unblock. **This is why `producer_commit` is a no-op for TMA-producer pipelines** (the TMA op itself signals). [VERIFIED — CUTLASS C++ pipeline doc symmetry rule]

- High-level: `pipeline.PipelineTmaUmma.create(..., tx_count=<bytes per stage>, ...)` does the `arrive_and_expect_tx` for you; `tx_count` = total TMA bytes landing in one stage's smem buffer. The producer's `acquire` internally calls `arrive_and_expect_tx(tx_count)`. [VERIFIED]
- Low-level (`cute.arch`): `mbarrier_init(mbar, cnt)`, `mbarrier_init_fence()`, `mbarrier_arrive_and_expect_tx(mbar, bytes, peer_cta_rank_in_cluster=None)`, `mbarrier_expect_tx(mbar, bytes, ...)` (expect, no arrive), `mbarrier_wait(mbar, phase)`, `mbarrier_try_wait(mbar, phase) -> Boolean`. [VERIFIED]
- **Extra/variable tx bytes** (a side load piggy-backing on the same barrier): add via `mbarrier_expect_tx`, or use FA4's `producer_acquire(..., extra_tx_count=N)` override (`pipeline.py:304`) which sets `tx_count = sync_object_full.tx_count + extra_tx_count`. (No `extra_tx_count` constructor param exists — it is an override/`mbarrier_expect_tx` mechanism.) [VERIFIED FA4 src]

### 4.5 TMA — SMEM layout + swizzle

SMEM layouts feeding TMA+UMMA must be legal for **both** simultaneously. Build via the swizzle-atom helpers, never a plain `make_layout`:
```python
atom = tcgen05.make_smem_layout_atom(kind, element_type)            # kind = SmemLayoutAtomKind
sA   = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, a_dtype, num_stages)  # convenience
```
`SmemLayoutAtomKind` (sm100): `MN_INTER, MN_SW32, MN_SW64, MN_SW128, MN_SW128_32B, K_INTER, K_SW32, K_SW64, K_SW128`. [VERIFIED — `tcgen05` enum]
- `*_SW128` = **128-byte swizzle** (the common GEMM choice; max bank-conflict avoidance for a 128B cache line).
- `MN_SW128_32B` = 128B swizzle with **32-byte atomicity** — required for **transposed / MN-major** operands (the MMA docstrings: transpose only with 128B swizzle + 32B swizzle-atomicity).
- `*_INTER` = interleaved/no-swizzle; `*_SW32/64` = smaller swizzles for narrower tiles.
- A ComposedLayout returned by these = layout **+** swizzle: `.outer` = the plain layout (pass to `allocate_tensor(layout=...)`), `.inner` = the swizzle (pass to `swizzle=...`). [VERIFIED — cheatsheet §2]
- C++ name for `*_SW128` is `UMMA::Layout_K_SW128_Atom<T>` (the "NVMMASharedLayout / 128B swizzle" of the Gluon world is this same UMMA swizzled layout). [VERIFIED — tutorial `04_mma_tma_2sm_sm100.cu`]

**Manual-fill caveat (project, B200-validated):** you can hand-fill a swizzled MMA operand by **logical** coordinate and tcgen05 consumes it correctly — `sX[(row, ki), 0, kb, 0] = val` (with `k = kb*8 + ki`). TMA is NOT required to feed tcgen05; the descriptor only cares about the resulting smem layout, not how the bytes arrived. (refutes "swizzled coord-write silently corrupts".) [project memory `cutedsl-m3c-validated-primitives`]

### 4.6 tcgen05 copy — TMEM read-back / write-back

The accumulator lives in TMEM; you move it to/from registers with dedicated `tcgen05.ld`/`tcgen05.st` atoms tiled by `tcgen05.make_tmem_copy`.

| Atom | `(num_dp, num_bits)` | Notes |
|---|---|---|
| `Ld32x32bOp` / `St32x32bOp` | (32, 32) | 32 data-paths × 32-bit — the workhorse (≈ C++ `SM100_TMEM_LOAD_32dp32b`) |
| `Ld16x64bOp` / `St16x64bOp` | (16, 64) | |
| `Ld16x128bOp` / `St16x128bOp` | (16, 128) | |
| `Ld16x256bOp` / `St16x256bOp` | (16, 256) | widest |
| `Ld16x32bx2Op` / `St16x32bx2Op` | (16, 32)×2 | |

Each takes `repeat: tcgen05.Repetition` (`x1…x128`) and `pack: tcgen05.Pack` (`PACK_16b_IN_32b` or `NONE`). [VERIFIED — `tcgen05/copy.py`, `get_tmem_copy_properties`]

**`make_tmem_copy`** builds the tiled copy; **`sm100_utils.get_tmem_load_op(...)`** auto-selects the best load atom. Canonical TMEM→RMEM (t2r) read-back — `sm100_hd256_2cta_fmha_forward.py:1547`:
```python
tmem_load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), acc_dtype)
tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)   # tile by the TMEM tensor
thr_load = tmem_tiled_load.get_slice(thread_idx)
tTMEM_LOADtS = thr_load.partition_S(tStS_slice)                        # TMEM source
tTMEM_LOADrS = cute.make_rmem_tensor(thr_load.partition_D(tScS_slice).shape, acc_dtype)
cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)                 # TMEM -> regs
cute.arch.fence_view_async_tmem_load()                                 # MANDATORY before reuse/release
s_handle.release()
```
RMEM→TMEM (r2t) write-back uses `St32x32bOp`, then `fence_view_async_tmem_store()` before signalling — `sm100_hd256_2cta_fmha_forward.py:1586`. The `dense_gemm.py` epilogue uses `sm100_utils.get_tmem_load_op(...)` + `make_tmem_copy` to pick the atom automatically. [VERIFIED]

**SMEM→TMEM (s2t, for the `a_src=OperandSource.TMEM` path):** `tcgen05.make_s2t_copy(...)` + `tcgen05.get_s2t_smem_desc_tensor(...)`. [VERIFIED — `tcgen05/__init__.py`]

**Read TMEM back in FP32** — never read a tf32 accumulator directly as output (it rounds to ~19 mantissa bits). The acc dtype is F32; store-convert to the output dtype on the way out. [project memory + tf32x3 floor]

### 4.7 `cute.copy` — the universal verb

`cute.copy(atom_or_tiled_copy, src, dst, **kw)` drives **every** mover above; the atom decides the mechanism:
- TMA: `cute.copy(tma_atom, gmem_view, smem_view, tma_bar_ptr=bar[, mcast_mask=mask])`
- cp.async / universal: `cute.copy(copy_atom, src, dst[, pred=pred])` (then `cp_async_commit_group()`/`cp_async_wait_group(n)`)
- TMEM ld/st: `cute.copy(tiled_tmem_copy, tmem_src, rmem_dst)` (and reverse)
- vectorized smem→reg shortcut: `cute.autovec_copy(src, dst)` (`copy_utils.py:35` `load_s2r`)

`elect_one` caveat: do **NOT** wrap `cute.copy(TMA...)` in `cute.arch.elect_one()` — the TMA path self-elects; wrapping deadlocks. (Same rule as `cute.gemm`.) [cheatsheet §3]

### 4.8 Fences — which one, when (the correctness-critical part)

These are **mandatory** ordering fences between an async producer and its consumer; omitting them is a silent-corruption / race bug, not a perf nit. All under `cute.arch`.

| Fence | Use after… | …before | Producer kind |
|---|---|---|---|
| `fence_view_async_shared()` | writing **SMEM** (r2s / st.shared) | signalling a pipeline / arriving a barrier so an async consumer (TMA store **or** UMMA) reads it | **SMEM producer** (compute warp filling an operand) |
| `fence_view_async_tmem_store()` | writing **TMEM** (r2t, `St32x32b`) | signalling that the TMEM write is visible to the MMA / another warp | **TMEM producer** |
| `fence_view_async_tmem_load()` | reading **TMEM**→registers (t2r) | reusing those registers / **releasing the acc stage** | TMEM **consumer** (orders the read before the slot is freed) |
| `fence_proxy("async.shared", space="cta")` | generic async-proxy ordering with explicit space | crossing the async proxy (alt spelling of the smem fence) | generic |

Decision rule:
- **SMEM producer → consumer:** `fence_view_async_shared()`. FA4 softmax→MMA hand-off — `flash_bwd_mla_sm100.py:1885`:
  ```python
  cute.copy(tiled_copy_r2s, rPt_copy_view, tSR_sPt_cur)   # regs -> smem
  cute.arch.fence_view_async_shared()                     # visible to async proxy
  self.softmax_barrier.arrive_and_wait()
  pipeline_Pt.producer_commit(producer_state_Pt)          # now signal the MMA warp
  ```
- **TMEM producer (you just wrote the acc / stats):** `fence_view_async_tmem_store()` before the commit — `flash_fwd_sm100.py:2353`.
- **TMEM consumer (you just read the acc out):** `fence_view_async_tmem_load()` before releasing the stage — `sm100_hd256_2cta_fmha_forward.py:1557`.
- Other generic fences (`cute.arch`): `fence_proxy(kind, space)` (kinds: `alias/async/async.global/async.shared/tensormap/generic`), `fence_acq_rel_{cta,cluster,gpu,sys}()`, `mbarrier_init_fence()`. [VERIFIED]

**Critical pairing with manual-fill → tcgen05 (the QR apply path):** when a compute warp hand-fills a swizzled SMEM operand and a tcgen05 MMA consumes it, the producer must `fence_view_async_shared()` after the `st.shared` writes and **before** arriving the full-barrier. Because the producer is an async *thread* (not a TMA), its `producer_commit` is a **real `mbarrier.arrive`** (use `PipelineAsyncUmma`, or hand-rolled mbarriers) — unlike the TMA case where commit is a no-op. (See pipeline section for the full ring recipe + the `tcgen05.commit(mbar)` UMMA-completion arrive that frees the stage.) [VERIFIED mechanism + project memory `cutedsl-m3c-validated-primitives` 13b]

---

## 5. MMA & tensor cores (Blackwell sm100 / tcgen05)

> The Blackwell tensor core is **`tcgen05`** (a.k.a. UMMA). Three things make it different from Hopper `wgmma`: the **accumulator lives in TMEM**, not registers; MMAs come in **1-SM (`CtaGroup.ONE`) and 2-SM (`CtaGroup.TWO`)** flavors; and `cute.gemm` issuing a `tcgen05.mma` is **asynchronous** — completion is observed via `tcgen05.commit` on an mbarrier. All atoms in this section are **sm100-only** (`sm_100a/f`, `sm_103a/f`, `sm_110a/f`); they do not exist on sm90.

Module aliases used below:
```python
import cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, cpasync
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.pipeline as pipeline
```

---

### 5.1 The MMA Op classes

`tcgen05.__all__` exports (verbatim, `tcgen05/mma.py`): `MmaTF32Op`, `MmaF16BF16Op`, `MmaI8Op`, `MmaFP8Op` (deprecated → `MmaF8F6F4Op`), `MmaF8F6F4Op`, and block-scaled `MmaMXF8Op / MmaMXF8F6F4Op / MmaMXF4Op / MmaMXF4NVF4Op`.

| Op | A/B dtype | Acc | **Inst-K (fixed)** | Note |
|---|---|---|---|---|
| `MmaTF32Op` | TF32 | **F32** | **8** | inst-K MUST be 8 (raises otherwise) |
| `MmaF16BF16Op` | F16 | F16 **or F32** | **16** | BF16 → F32 acc only |
| `MmaF8F6F4Op` | E4M3/E5M2 | F16 or F32 | **32** | A/B dtype must match |
| `MmaI8Op` | int8/uint8 | int32 | 32 | |

Constructor signatures (VERIFIED, `mma.py`):
```python
MmaTF32Op(instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)
MmaF16BF16Op(ab_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)
```
Real call (this repo, `cute_gemm_tf32x3.py:148`):
```python
op = tcgen05.MmaTF32Op((128,256,8), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                       tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
tiled_mma = cute.make_tiled_mma(op)
```

**Enums** (VERIFIED):
- `OperandSource.SMEM` (default; A from an SMEM matrix descriptor) / `OperandSource.TMEM` (A read directly from TMEM — and then **A is always K-major**; transposed-A requires `SMEM`).
- `CtaGroup.ONE` / `CtaGroup.TWO` (1-SM vs 2-SM paired-CTA MMA).
- `cute.nvgpu.OperandMajorMode.K` / `.MN`. ⚠️ `tcgen05.OperandMajorMode` is **deprecated** — use `cute.nvgpu.OperandMajorMode`. (The repo snippet above still uses `tcgen05.OperandMajorMode.K` and works on 4.5.2; prefer the `cute.nvgpu` spelling going forward.)
- `tcgen05.Field`: `ACCUMULATE` (`"accum_c"`), `NEGATE_A`, `NEGATE_B`, `SFA`, `SFB`, `DISABLE_OUTPUT_LANE` — runtime-settable via `tiled_mma.set(field, value)`.

**Instruction-shape constraints** (identical across TF32/F16BF16/FP8 docstrings):
- `CtaGroup.ONE`: M ∈ {64, 128}; 8 ≤ N ≤ 256 step 8.
- `CtaGroup.TWO`: M ∈ {128, 256}; 16 ≤ N ≤ 256 step 16.
- Transpose (MN-major A/B) only with **128B swizzle, 32B swizzle-atomicity** (`SmemLayoutAtomKind.MN_SW128_32B`).

C++ atom names the DSL wraps (so you can cross-read CUTLASS C++/Gluon): `SM100_MMA_F16BF16_SS<...>` (1-SM, `_SS` = both operands from SMEM), `SM100_MMA_F16BF16_2x1SM_SS<...,256,256,...>` (2-SM), `_TS` = A-from-TMEM. The 7 PTX kinds: `.kind::{tf32,f16,i8,f8f6f4}` + block-scale `{mxf8f6f4,mxf4,mxf4nvf4}`.

---

### 5.2 make_tiled_mma / make_trivial_tiled_mma

Two paths, both valid:
- **Generic:** `cute.make_tiled_mma(op_instance)` — pass an `MmaTF32Op(...)` etc. (used in `cute_gemm_tf32x3.py:150`).
- **Helper (recommended for full GEMM):** `sm100_utils.make_trivial_tiled_mma(...)` — derives MMA tiling + cluster wiring for you. FA4, `sm100_hd256_2cta_fmha_forward.py:402`:
```python
qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a_dtype, a_major, b_major, acc_dtype, cta_group, mma_tiler_mn[:2])
# A-from-TMEM variant (P-operand lives in TMEM):
pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
    v_dtype, OperandMajorMode.K, v_major, pv_acc_dtype, cta_group, pv_tiler[:2],
    tcgen05.OperandSource.TMEM)
```
Full helper signature (VERIFIED): `make_trivial_tiled_mma(a_dtype, b_dtype, a_leading_mode, b_leading_mode, acc_dtype, cta_group, mma_tiler_mn, a_source=OperandSource.SMEM)`. The legacy single-`ab_dtype` overload is deprecated.

---

### 5.3 cute.gemm — accumulate semantics & the K-loop

**The accumulator is a TMEM tensor**, not a register fragment. Canonical setup (VERIFIED, `dense_gemm.py` + this repo):
```python
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
tCtAcc   = tiled_mma.make_fragment_C(acc_shape)            # LAYOUT only (fake)
tmem.wait_for_alloc()                                       # MANDATORY before retrieve_ptr
tCtAcc   = cute.make_tensor(tmem.retrieve_ptr(acc_dtype), tCtAcc.layout)  # real TMEM tensor
tCrA = tiled_mma.make_fragment_A(sA)                       # SMEM-desc fragments
tCrB = tiled_mma.make_fragment_B(sB)
```

**ACCUMULATE flag = the C-scale.** The FIRST `cute.gemm` of a fresh accumulator runs with `ACCUMULATE` unset (overwrite, C++ `UMMA::ScaleOut::Zero` — C is ignored); set it `True` afterward so subsequent K-blocks add (`ScaleOut::One`). Argument order is **`cute.gemm(tiled_mma, D, A, B, C)`** with D and C the same TMEM tensor. This repo, `cute_gemm_tf32x3.py:128`:
```python
for kb in cutlass.range_constexpr(nkb):
    cute.gemm(tiled_mma, tCtAcc, tCrAhi[kc], tCrBhi[kc], tCtAcc)   # k=0: overwrite
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)                  # all later: accumulate
    cute.gemm(tiled_mma, tCtAcc, tCrAhi[kc], tCrBlo[kc], tCtAcc)
    cute.gemm(tiled_mma, tCtAcc, tCrAlo[kc], tCrBhi[kc], tCtAcc)
```
The FA4 helper `blackwell_helpers.py:96` packages the toggle as a `zero_init` flag:
```python
@cute.jit
def gemm(tiled_mma, acc, tCrA, tCrB, zero_init=False):
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
        cute.gemm(mma_atom, acc, tCrA[None,None,k], tCrB[None,None,k], acc)
```

**Execution-model rules (do not get these wrong):**
- `cute.gemm` (→ `tcgen05.mma`) is **async** and must be issued **warp-uniform**. **Do NOT wrap it in `elect_one()`** — the compiler handles single-thread issue; wrapping deadlocks. (Same rule as `cute.copy(TMA...)`.)
- To make a dependent reader (epilogue, next stage) observe completion, signal an mbarrier with **`tcgen05.commit(mbar_ptr, mask, cta_group)`** — and this **MUST** be `elect_one`-guarded (else 32× redundant commits → phase desync → deadlock). This is the single most load-bearing primitive for hand-rolled rings (see §5.6). `tcgen05.commit(mbar, mask=None, cta_group=ONE)` is exactly what `PipelineUmmaAsync.producer_commit` calls internally.
- TMEM ld/st completion: PTX `tcgen05.wait::ld` / `wait::st`. Cross-thread ordering around TMEM: `cute.arch.fence_view_async_tmem_store()` after a TMEM write (before signalling), `cute.arch.fence_view_async_tmem_load()` after reading TMEM into registers (before reuse/release). After an SMEM write that UMMA will read: `cute.arch.fence_view_async_shared()`.

---

### 5.4 The MMA descriptor (mma_sm100_desc)

You normally **do not** hand-build a descriptor — `make_fragment_A/B` over a correctly-swizzled SMEM layout produces the 64-bit UMMA **SMEM matrix descriptor** for you (for `OperandSource.SMEM`). For `OperandSource.TMEM`, A is read straight from a TMEM tensor (K-major). DSL helpers that surface the descriptor exist (`tcgen05.make_umma_smem_desc`, `int_to_smem_descriptor`/`smem_descriptor_to_int`) but are rarely needed.

When the high-level path isn't enough, FA4 hand-rolls the raw `tcgen05.mma` inline-PTX path. The descriptor packers (for that path only):
- 32-bit **instruction descriptor** `idesc`: `make_instr_desc` (`mma_sm100_desc.py:111`); `mma_op_to_idesc(op)` (`mma_sm100_desc.py:165`) converts an `MmaOp` directly.
- 64-bit **SMEM descriptor**: `make_smem_desc_base` (`mma_sm100_desc.py:212`) + start-addr `make_smem_desc_start_addr` (`mma_sm100_desc.py:285`) — encodes LBO/SBO/swizzle mode.
- The inline asm itself: `blackwell_helpers.py:201`, `tcgen05.mma.cta_group::1.kind::{kind} [$0], smem_desc_a, smem_desc_b, idesc, p;` with `kind` from `_tcgen05_mma_kind` (`blackwell_helpers.py:13`).

> The SMEM descriptor only cares about the **smem layout**, not how the bytes got there — so a manually `st.shared`-filled (transposed/masked) operand is a legal MMA input as long as you match the swizzled layout. (Validated on B200 in this repo: hand-filling `sX[(row,ki),0,kb,0]` works; refutes "swizzled coord-write silently corrupts".)

---

### 5.5 dtypes + FP32 accumulation, and TMEM accumulators across a K-loop

- **FP32 input → TF32 MMA via TMA:** pass `internal_type=cutlass.TFloat32` to `make_tiled_tma_atom_A/B` so an FP32 GMEM tensor lands as TF32 for the MMA. `MmaTF32Op` forces operands→TFloat32 and acc→Float32.
- **FP16/BF16/FP8 all accumulate in FP32** (FP16 may use F16 acc but you almost never want to). This is why **fp16x3 is *not* a speed win over tf32x3** on this hardware: fp16 with fp32-accumulate runs at the TF32 rate.
- **TMEM accumulator persists across the K-loop.** You allocate the TMEM tensor once, run the whole K-loop of `cute.gemm` calls accumulating into it, then read it back. The accumulator is live in TMEM the entire loop — no register spill.

**TMEM budget / the occupancy wall** (VERIFIED, `cute/arch/tmem.py`): max alloc = **512 columns** (`TMEM_MAX_ALLOC_COLUMNS_MAP["sm_100"] == 512`); min = 32; alloc count must be **a multiple of 32 AND a power of two**. Geometry is 128 lanes × 512 cols of 32-bit ≈ 256 KB/SM. A `[128,256]` F32 accumulator = 256 columns; a full 512-column acc monopolizes TMEM → **1 CTA/SM**. This is the "NBP≥64 / 1-CTA-per-SM" wall behind the project's ~2.0× serial-tcgen05 floor at n≤512: TMEM is a single per-SM resource, so a big accumulator forecloses co-residency. To get >1 CTA/SM, shrink the acc (fewer columns) or sub-allocate via `TmemBufferPool`.

**Allocate / read-back** (VERIFIED): `tmem = utils.TmemAllocator(...)`; `tmem.allocate(ncols)` (allocator warp only) → `tmem.wait_for_alloc()` (NamedBarrier so all warps see the base ptr — **must precede `retrieve_ptr` in every warp that reads acc**, else garbage pointer → fault) → `tmem.retrieve_ptr(Float32)`. TMEM→RMEM uses `tcgen05.ld` atoms (`Ld32x32bOp`, `Ld16x{64,128,256}bOp`) built with `tcgen05.make_tmem_copy(atom, tmem_tensor)`; the performant atom is auto-picked by `sm100_utils.get_tmem_load_op(...)`. Read the acc back in **FP32** (never read a tf32 acc as the output dtype — it rounds to ~19 bits). Repo readback, `cute_gemm_tf32x3.py:106`:
```python
tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
...
cute.copy(tmem_tiled_copy, tDtC[None,None,i], tCrAcc)   # TMEM -> regs
tCrC.store(tCrAcc.load().to(io_dtype))                  # convert-on-store, FP32-safe
```

---

### 5.6 tf32x3 emulation — the in-register split idiom (read this carefully)

There is **no native tf32x3 MMA op.** "3×TF32" is an *emulation* on top of `MmaTF32Op`: split each FP32 value into a TF32 **hi** limb and a TF32 **lo** (residual) limb, then issue **3 TF32 MMAs accumulated in one FP32 TMEM acc** — `hi·hi + hi·lo + lo·hi` (the `lo·lo` cross-term is dropped). This recovers ~22 mantissa bits at ~the TF32 rate. (CUTLASS calls this "3xTF32"; the original is the Ampere `27_ampere_3xtf32` example.) The 3-pass loop is exactly §5.3 above. Use 6 passes only if a tighter margin is needed; gate dropping `lo·lo` on the mixed@640 accuracy margin (≈1.83 was observed sufficient).

**The split.** Host-side (this repo, `cute_gemm_tf32x3.py:166`, validated): truncate the low 13 mantissa bits to get the TF32 hi limb, residual is lo:
```python
hi = (x.view(torch.int32) & (~0x1FFF)).view(torch.float32)   # clear low 13 mantissa bits
lo = x - hi
```

**In-register split (B200-validated, each alternative cost a Modal run):**
```python
xi = x.bitcast(cutlass.Int32)
hi = (xi & cutlass.Int32(-8192)).bitcast(cutlass.Float32)    # -8192 == 0xFFFFE000, clears low 13 bits
lo = x - hi
```
This is validated EXACT (maxabs 0.0; GEMM-1 rel 8.25e-7). **Three traps — all of these are WRONG:**

| Wrong idiom | Why it fails |
|---|---|
| `cute.bitcast(x, ...)` | **`cute.bitcast` does not exist** (a build-guide line claiming it is a bug). Bitcast is a **method**: `value.bitcast(dtype)`. (Tensor-level reinterpret is `cute.recast_tensor`.) |
| `cutlass.and_(xi, ...)` / `cute.core.and_(...)` | These are **logical-AND**, not bitwise → produce **NaN**. Use the **Python `&`** operator on the bitcast `Int32` value. |
| Dekker split `c-(c-x)` **or** `x.to(TFloat32).to(Float32)` round-trip | **Both collapse to `x` under the optimizer** — rel stays stuck at ~8.8e-4 (= plain 1×tf32, no improvement). **Only the bit-AND survives optimization.** |

So: **`(x.bitcast(Int32) & Int32(-8192)).bitcast(Float32)`** — Python `&`, value-method `.bitcast`, magic constant `-8192`. Nothing else works on 4.5.2. *(All three traps are project-empirical on B200; the underlying `.bitcast`/`recast_tensor` API is VERIFIED in docs — verify the exact behavior on your wheel if it drifts.)*

---

### 5.7 The async ring: feeding a UMMA consumer from a manually-filled SMEM operand

If your MMA operand is **hand-filled** (transposed/masked V, WY blocks — i.e. not TMA-loaded), the high-level pipeline classes have a fatal trap:

> ⚠️ **`PipelineTmaUmma.producer_commit` (and the TMA-style commit) is literally a `pass` no-op** — "TMA producer commit is a noop since the TMA instruction itself updates the transaction count." Umma pipelines arrive the full-barrier via the **async-copy instruction**, NOT via `producer_commit`. So a manual `st.shared` fill **never arrives the barrier** → the MMA consumer waits forever → **deadlock.** (The Python-DSL docstrings even mislabel `PipelineUmmaAsync`/`PipelineAsyncUmma` commit as the TMA no-op — that's a copy-paste doc artifact; trust the C++ symmetry rule.)

For an async-thread→UMMA handoff the *intended* class is **`PipelineAsyncUmma`** (producer = async thread → `producer_commit` is a **real** `mbarrier.arrive`; consumer = UMMA). FA4 uses it for the softmax-writes-P → MMA path (`sm100_hd256_2cta_fmha_forward.py:635`; commit at `flash_bwd_mla_sm100.py:1885` after `fence_view_async_shared()`). **But** in this repo the first attempts deadlocked, and the validated production fix was a **hand-rolled low-level mbarrier ring** driven by `tcgen05.commit`:

**The validated ring recipe (B200, 1.58× faster than a per-tile CTA barrier):**
- SharedStorage: `full_mbar[stages]`, `empty_mbar[stages]`, `acc_done[1]` (all `MemRange[Int64]`).
- **init (tidx==0 only):** `mbarrier_init(full+s, NPROD)`, `mbarrier_init(empty+s, 1)`, `mbarrier_init(accd, 1)`; **PRIME** each `empty+s` with `mbarrier_arrive`; `mbarrier_init_fence()`; one `cute.arch.barrier()` (one-time).
- **PRODUCER warps:** per k-tile → `mbarrier_wait(empty+s, ph)`; fill the swizzled smem operand for stage `s`; `cute.arch.fence_view_async_shared()`; `mbarrier_arrive(full+s)`.  (`s = kt%2`, `ph = (kt//2)%2` for 2 stages.)
- **CONSUMER (MMA) warp:** per k-tile → `mbarrier_wait(full+s, ph)`; 3-pass `cute.gemm` ACC-accumulate into TMEM; **`if lane_idx()==0: tcgen05.commit(empty+s)`** ← UMMA-completion frees the stage. **ELECT-ONE is mandatory** (all-32 over-arriving a count-1 barrier → phase desync → deadlock). After the loop: `if lane0: tcgen05.commit(accd)`, then all 128 threads `mbarrier_wait(accd, 0)` → readback.

**Phase bookkeeping gotchas:** 2 stages → safe binary phase XOR; >2 stages needs mod-N phase. The arrive-count `NPROD` must **exactly** match the number of producer arrives. And the multi-stage heisenbug to avoid: **do NOT re-init mbarriers inside nested loops** (per-stripe re-init races the ring's async phase → hangs without a `cute.printf`, "passes" with one because the print delay masks the race). **Fix = init mbarriers ONCE + carry running-parity phases across the nested loops, never re-init.**

**Tracing limitation that bites here:** nested `def`s capturing pipeline objects fail with *"closures not supported in dynamic control flow"* — **inline the fill / MMA / readback bodies**. And the stage index in a fragment slice is mandatory: `(None, None, kb, stage)`.

---

### 5.8 Quick rules summary

- `cute.gemm` = async, warp-uniform, **no `elect_one`**. `tcgen05.commit` = **needs `elect_one`**. Mixing these up is the classic 32×-commit deadlock.
- First MMA into a fresh acc: `ACCUMULATE=False` (overwrite); all subsequent: `True`.
- Acc lives in **TMEM**; 512 cols max, multiple-of-32 & power-of-two; a full-width acc = 1 CTA/SM (the tcgen05 occupancy wall).
- Read the acc back in **FP32** via `tcgen05.ld` (`Ld32x32bOp`) + `make_tmem_copy`; convert-on-store to the output dtype.
- `tmem.wait_for_alloc()` before every `retrieve_ptr`; fence TMEM (`fence_view_async_tmem_store/load`) and SMEM (`fence_view_async_shared`) around producer/consumer handoffs.
- tf32x3 split = **`(x.bitcast(Int32) & Int32(-8192)).bitcast(Float32)`** — Python `&`, NOT `cutlass.and_`; `.bitcast` is a method, NOT `cute.bitcast`; Dekker and `.to()` round-trips collapse under the optimizer.
- 2-SM path = `CtaGroup.TWO` + `TmemAllocator(is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=...)` + `make_tiled_tma_atom_*` with a cluster. *(end-to-end 2-SM wiring: verify on B200)*

---

## 6. Pipelines & synchronization (the core of warp-spec)

> This is THE section. Warp-specialization in cute-DSL 4.5.2 = manual warp-id roles (there is **no `warp_specialize` op** — that's a Gluon-ism) + the `cutlass.pipeline` class family doing producer/consumer mbarrier handshakes between those roles. Get the pipeline-class choice wrong and you **deadlock silently**.

### 6.0 The mental model: two mbarriers per stage

A pipeline = a ring of `num_stages` smem slots, each guarded by **two mbarriers** [VERIFIED, Colfax / NVIDIA C++ doc]:

- **`full_barrier`** — "data ready". Producer arms it; consumer waits on it.
- **`empty_barrier`** — "slot free to overwrite". Consumer arms it; producer waits on it.

The four methods map onto those two barriers:

| Method | Acts on | Blocking? | Meaning |
|---|---|---|---|
| `producer_acquire(state)` | waits `empty_barrier` | **yes** | wait until consumer freed this slot |
| `producer_commit(state)` | arms `full_barrier` | no | data is ready |
| `consumer_wait(state)` | waits `full_barrier` | **yes** | wait until data is ready |
| `consumer_release(state)` | arms `empty_barrier` | no | done reading, slot free |

★ **The load-bearing rule** [VERIFIED, NVIDIA C++ pipeline doc]: *"if the Pipeline class is `PipelineTmaAsync`, then `full_barrier` is wrapped as a `ClusterTransactionBarrier` and the signaling mechanism is handled by the TMA load itself via incrementing the transaction count. In this case the `producer_commit` method is actually a no-op."* — **whether `producer_commit` does anything depends entirely on what kind of producer arms `full_barrier`.** This single sentence is the root cause of the deadlock in §6.4.

Docs: [C++ pipeline](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html), [Python DSL pipeline API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html), [Colfax GEMM-pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/).

### 6.1 ★ The class-selection table: producer-type × consumer-type → which class

Class name reads `Pipeline<Producer><Consumer>` (`Async` = an ordinary async thread). **Producer kind decides how `full_barrier` is armed; consumer kind decides how `empty_barrier` is armed.**

| Class | Producer (arms `full_barrier`) | Consumer (arms `empty_barrier`) | `producer_commit` is… | Use for |
|---|---|---|---|---|
| `PipelineAsync` | async thread | async thread | **REAL mbarrier arrive** | generic thread→thread smem handoff |
| `PipelineCpAsync` | `cp.async` | async thread | real arrive (after cp.async group commit/wait) | cp.async load → compute |
| `PipelineTmaAsync` | TMA load | async thread | **NO-OP** (TMA tx-count arms it) | Hopper mainloop: TMA → threads/WGMMA |
| `PipelineTmaUmma` | TMA load | UMMA (tcgen05) | **NO-OP** (TMA tx-count) | Blackwell mainloop: TMA → tcgen05 MMA |
| `PipelineUmmaAsync` | UMMA (tcgen05) | async thread | UMMA arrives via `tcgen05.commit` | TMEM accumulator → epilogue threads |
| `PipelineAsyncUmma` | **async thread** | UMMA (tcgen05) | **REAL mbarrier arrive** | **manual-filled / compute-produced smem operand → tcgen05 MMA** |

(C++ spells `PipelineTmaUmma` as `PipelineTmaUmmaAsync`. Other classes exist — `PipelineTmaMultiConsumersAsync`, `PipelineClcFetchAsync`, `PipelineTmaStore` — but the six above are the warp-spec core.)

**Directional summary — memorize this:**
- **TMA → anything:** `PipelineTmaAsync` (→threads) / `PipelineTmaUmma` (→UMMA). `producer_commit` = no-op; the TMA instruction itself arms `full_barrier` via transaction bytes.
- **UMMA → threads:** `PipelineUmmaAsync`. Producer "commit" is the MMA's own `tcgen05.commit`.
- **threads → UMMA:** `PipelineAsyncUmma`. Producer is an ordinary thread → `producer_commit` is a **real `mbarrier.arrive`**. ← §6.4, the manual-fill case.
- **threads → threads / cp.async → threads:** `PipelineAsync` / `PipelineCpAsync`, real arrives both ways.

⚠️ **Doc artifact [VERIFIED in the wheel]:** the Python docstrings for `PipelineUmmaAsync` *and* `PipelineAsyncUmma` both render `producer_commit` as *"TMA producer commit is a noop since TMA instruction itself updates the transaction count"* — a **copy-paste bug** from `PipelineTmaAsync`, **wrong** for the UMMA/async classes. The project read the actual source (`cute_pipesrc.py` introspection) and confirmed: `PipelineTmaUmma.producer_commit` body is literally `pass`, but the async-producer classes do a real arrive. **Trust the C++ symmetry rule + source over the Python docstring.**

### 6.2 `create()` signatures + `PipelineState`

```python
import cutlass.pipeline as pipeline
from cutlass.pipeline import CooperativeGroup, Agent, PipelineUserType

# tx_count exists ONLY on the TMA classes:
PipelineTmaUmma.create(num_stages, producer_group, consumer_group, tx_count,
                       barrier_storage=None, cta_layout_vmnk=None,
                       mcast_mode_mn=(1,1), defer_sync=False)
PipelineUmmaAsync.create(num_stages, producer_group, consumer_group,
                         barrier_storage=None, cta_layout_vmnk=None, defer_sync=False)   # NO tx_count
PipelineAsyncUmma.create(num_stages, producer_group, consumer_group,
                         barrier_storage=None, cta_layout_vmnk=None, defer_sync=False)   # NO tx_count
PipelineAsync.create(num_stages, producer_group, consumer_group,
                     barrier_storage=None, producer_mask=None, consumer_mask=None, defer_sync=False)
```

- `barrier_storage`: `cute.Pointer` into smem holding this pipeline's mbarrier array — you allocate it (`storage.<field>.data_ptr()`). [VERIFIED, FA4 `sm100_hd256_2cta_fmha_forward.py:607`]
- `producer_group` / `consumer_group`: `CooperativeGroup(Agent.Thread, size)` — **`size` = the exact arrival count for that side.** Mismatch = phase desync (§6.5). FA4 computes it as `len(warp_ids) * threads_per_warp * cluster_m` for a multi-warp producer. [VERIFIED, `sm100_hd256_2cta_fmha_forward.py:635`]
- `defer_sync=True`: defers the cluster-init barrier (common in warp-spec where the init barrier is issued elsewhere).
- `make_participants()` (FA4 ergonomic) splits one pipeline object into `(producer, consumer)` handles. [VERIFIED, `sm100_hd256_2cta_fmha_forward.py:607`]

**`PipelineState` (index / phase)** [VERIFIED]:
```python
state = pipeline.make_pipeline_state(PipelineUserType.Producer, num_stages)   # or .Consumer
state.index    # 0..num_stages-1  → which smem slot & which mbarrier
state.phase    # 1-bit; the phase to wait on (flips when a barrier completes)
state.count    # monotone counter (use to index gmem k-tiles, independent of the modular index)
state.advance()  # index++ (mod stages); flips phase on wrap
```
- The mbarrier holds `(pending-arrivals, expected-tx, phase-bit)`. When pending-arrivals **and** expected-tx both reach zero, the phase bit flips; waiters compare against the phase they captured. [VERIFIED, Colfax]
- **Producers start with the phase bit flipped to 1** so the first `producer_acquire` sees buffers as empty. [VERIFIED, FA4 `flash_fwd_sm100.py:1357`: `q_producer_phase = Int32(1)`]
- Keep **separate** producer-state, consumer-wait-state, and consumer-release-state and `.advance()` each independently — that staggering is what creates the overlap. [VERIFIED, Veitner]

### 6.3 Driving the canonical pipelines (handle style + manual style)

**TMA producer → UMMA consumer** [VERIFIED, FA4 `sm100_hd256_2cta_fmha_forward.py:947, 1088`]:
```python
# producer (TMA warp): producer_commit is a no-op — the cute.copy ARMS the barrier itself.
q_handle = load_q_producer.acquire_and_advance()        # = producer_acquire + advance
cute.copy(tma_atom_q, tQgQ[None, iter], tQsQ[None, q_handle.index],
          tma_bar_ptr=q_handle.barrier)                 # TMA carries expect-tx → arms full_barrier
# consumer (MMA warp):
k_handle = load_kv_consumer.wait_and_advance()          # = consumer_wait + advance
cute.gemm(qk_tiled_mma, tStS_slice, tSrQ[...], tSrK[...], tStS_slice)
k_handle.release()                                       # consumer_release → arms empty_barrier
```
At loop end the TMA producer drains in-flight stages: `load_q_producer.tail()`.

**`tx_count` / `extra_tx_count`** [VERIFIED + INFERRED]: `tx_count` (create arg) = total TMA bytes landing in one stage's smem buffer; compute via `cute.size_in_bytes(dtype, smem_layout)`. On acquire the TMA pipeline internally calls `arrive_and_expect_tx(tx_count)`; the TMA copy then decrements expected-tx as bytes land → `full_barrier` completes. There is **no `extra_tx_count` create param** — it's a per-call override on `producer_acquire` (FA4 subclass, `pipeline.py:340`) for piggy-backing a side load on the same mbarrier; for ad-hoc extra bytes use the low-level `mbarrier_expect_tx(mbar, bytes)` (§6.6).

**elect-one commit:** for a TMA producer the pipeline self-elects, but in warp-spec code the TMA is issued by one elected thread anyway (`if warp_idx==0: with cute.arch.elect_one(): cute.copy(...)`). For a **manual** `producer_commit` (non-TMA), you must elect-one yourself or over-arrive (§6.5). FA4's `elect_one_commit` knob (`pipeline.py:101`) wraps the arrive in `cute.arch.elect_one()`.

### 6.4 ★★★ THE LESSON: feeding a UMMA consumer from a MANUALLY `st.shared`-filled operand

This is the single most expensive thing the project learned. Setup: you hand-fill a swizzled MMA operand in smem (transposed / unit-lower-masked V, a WY block — anything not coming from TMA) and want tcgen05 to consume it.

**The trap:** you reach for `PipelineTmaUmma` (it's the "→UMMA" class) → **silent deadlock.** Cause: `PipelineTmaUmma.producer_commit` body is literally `pass` (TMA arrives via tx-count, not via the commit). Your manual `st.shared` writes **never arm `full_barrier`** → the MMA consumer's `consumer_wait` blocks forever. [VERIFIED, project `cute_pipesrc.py` introspection + memory #11/#12]

**The fix — use `PipelineAsyncUmma`** (async-thread producer → UMMA consumer): its `producer_commit` is a **real `mbarrier.arrive`** issued by the threads that filled smem.

```python
pipe = pipeline.PipelineAsyncUmma.create(
    num_stages=S,
    producer_group=CooperativeGroup(Agent.Thread, n_fill_threads),  # exact count!
    consumer_group=CooperativeGroup(Agent.Thread, 1),               # the MMA leader
    barrier_storage=storage.mma_pipe.data_ptr())
p = pipeline.make_pipeline_state(PipelineUserType.Producer, S)
c = pipeline.make_pipeline_state(PipelineUserType.Consumer, S)

# producer (fill threads):
pipe.producer_acquire(p)                          # wait empty_barrier (slot free)
#   ... st.shared writes into the swizzled operand at slot p.index (apply V-mask during fill) ...
cute.arch.fence_view_async_shared()               # MANDATORY: make smem writes visible to async proxy
pipe.producer_commit(p)                           # REAL arrive on full_barrier
p.advance()

# consumer (MMA warp):
pipe.consumer_wait(c)                             # wait full_barrier
cute.gemm(tiled_mma, acc, sA[c.index], sB[c.index], acc)   # tcgen05.mma
pipe.consumer_release(c, cta_group=cta_group)     # "UMMA consumer release ... cta_group needs to be provided"
c.advance()
```

The exact production form is FA4's compute-warp→MMA handoff [VERIFIED, `flash_bwd_mla_sm100.py:1885`]:
```python
cute.copy(tiled_copy_r2s, rPt_copy_view, tSR_sPt_cur)   # registers -> smem
cute.arch.fence_view_async_shared()                     # visible to the async proxy
self.softmax_barrier.arrive_and_wait()
pipeline_Pt.producer_commit(producer_state_Pt)          # signal the MMA warp P is ready
producer_state_Pt.advance()
```

Why manual-fill is even legal: tcgen05 only cares about the **smem layout** the descriptor encodes (LBO/SBO/swizzle), not how the bytes got there — TMA is not required. The project validated hand-filling the swizzled operand by logical coord `sX[(row,ki),0,kb,0] = val` on B200 (refutes "swizzled coord-write silently corrupts"). [gau-nernst; project memory M3c #2]

**Ordering recipe (do not reorder):** `st.shared` fill → (CTA barrier so all writers done) → `fence_view_async_shared()` → `producer_commit` … consumer: `consumer_wait` → `cute.gemm` → `consumer_release(cta_group=...)`. If you fill via `cp.async` instead of plain `st.shared`, use `PipelineCpAsync` and add `cp_async_commit_group()`+`cp_async_wait_group()` before the arrive.

### 6.5 ★ The hand-rolled low-level mbarrier ring (the validated 1.58× recipe)

When the canned classes fight you (the `PipelineAsyncUmma` path is less-traveled), drop to raw mbarriers. NVIDIA explicitly endorses this [CUTLASS issue #2418]. The project's `cute_ring_g1.py` did exactly this and **beat the per-tile-CTA-barrier K-loop by 1.58×** (M=512: 69.8ms vs 110.2ms, both correct). Re-deriving it independently re-discovered the §6.4 lesson: a manual fill needs a real arrive, and `tcgen05.commit` is the UMMA-completion arrive.

**SharedStorage:** `full_mbar[stages]`, `empty_mbar[stages]`, `acc_done[1]` — all `MemRange[Int64]`.

**Init (tidx==0 only):**
```python
mbarrier_init(full + s,  NPROD)     # NPROD = exact producer arrival count (e.g. 96 = 3 warps×32)
mbarrier_init(empty + s, 1)         # one consumer arrives empty
mbarrier_init(accd, 1)
mbarrier_arrive(empty + s)          # PRIME: stages start empty so first producer_acquire passes
mbarrier_init_fence()
cute.arch.barrier()                 # ONE-TIME only — never re-init mbarriers inside the loop (§6.7)
```

**Producer (warps 0–2), per k-tile `kt`** with `s = kt % stages`, `ph = (kt // stages) % 2`:
```python
mbarrier_wait(empty + s, ph)        # slot free
#   ... fill smem[..., s] ...
cute.arch.fence_view_async_shared()
mbarrier_arrive(full + s)           # all NPROD threads arrive
```

**Consumer (warp 3), per k-tile:**
```python
mbarrier_wait(full + s, ph)
cute.gemm(...)                       # 3-pass tf32x3 ACC-accumulate into TMEM
if cute.arch.lane_idx() == 0:        # ★ ELECT-ONE MANDATORY
    tcgen05.commit(empty + s)        # UMMA-completion → frees the stage
# after the loop:
if cute.arch.lane_idx() == 0:
    tcgen05.commit(accd)
# then ALL 128 threads:
mbarrier_wait(accd, 0)
#   ... TMEM readback ...
```

★ **`tcgen05.commit(mbar, mask=None, cta_group=ONE)` is THE missing primitive** — it arrives on `mbar` when the async UMMA group completes (this is exactly what `PipelineUmmaAsync.producer_commit` calls internally). Reaches the empty-barrier *only after* the MMA has finished reading the operands, so the producer may safely overwrite the slot.

★ **elect-one is MANDATORY on `tcgen05.commit`:** if all 32 lanes arrive a count-1 barrier, the barrier over-arrives → phase desync → **deadlock**. (Same reason `cute.gemm` must NOT be wrapped in `elect_one` but `tcgen05.commit` MUST be — see §4.) [VERIFIED, project memory #13b; CUTLASS issue #2418 endorses the hand-roll]

⚠️ **The arrive/wait-on-different-barriers rule** [CUTLASS issues #2404/#2418]: a bare `mbarrier_arrive` + `mbarrier_wait` on the **same** barrier deadlocks — the arrive that satisfies the count flips the phase, so your own wait then waits on the *next* phase. The full/empty split (exactly what the pipeline classes do) is mandatory.

### 6.6 Low-level mbarrier + NamedBarrier + tcgen05 reference

`cutlass.cute.arch` primitives [VERIFIED signatures]:
```python
mbarrier_init(mbar_ptr, cnt)                     # arrival count
mbarrier_init_fence()                            # fence over the inits
mbarrier_arrive(mbar_ptr, peer_cta_rank_in_cluster=None)
mbarrier_arrive_and_expect_tx(mbar_ptr, bytes, peer_cta_rank_in_cluster=None)
mbarrier_expect_tx(mbar_ptr, bytes, ...)         # expect bytes, no arrive
mbarrier_wait(mbar_ptr, phase)                   # blocking
mbarrier_try_wait(mbar_ptr, phase) -> Boolean    # non-blocking
mbarrier_conditional_try_wait(cond, mbar_ptr, phase) -> Boolean
```

**Phase semantics** [VERIFIED]: when pending-arrivals AND expected-tx both hit zero, the phase completes and the mbarrier "immediately moves on to the next (incomplete) phase" — the bit flips. With `num_stages == 2` a binary phase XOR is safe; **>2 stages needs mod-N counting** (track `phase = (kt // stages) % 2` per slot, not a global flip).

**`tcgen05.commit` (UMMA-completion arrive)** [VERIFIED, gau-nernst + project]: PTX `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];` (C++ `cutlass::arch::umma_arrive` / `umma_arrive_multicast`). Groups prior async `tcgen05.mma`s and arrives the mbarrier on completion. **Must be elect-one** (§6.5). In the DSL it's reachable through the tcgen05 op wrappers / `cute.arch` (exact DSL spelling: verify on the wheel — the project calls it as `tcgen05.commit(mbar)`).

**Fences** — all four are MANDATORY correctness fences, easy to forget [VERIFIED, FA4]:
| Fence | When |
|---|---|
| `cute.arch.fence_view_async_shared()` | after `st.shared` / r2s smem write, **before** arming a pipeline an async consumer reads (TMA store or UMMA) |
| `cute.arch.fence_view_async_tmem_store()` | after writing TMEM (r2t), before signalling the consumer |
| `cute.arch.fence_view_async_tmem_load()` | after reading TMEM into registers (t2r), before reusing those regs / releasing the stage |
| `cute.arch.fence_proxy("async.shared", space="cta")` | generic async-proxy fence with explicit space |

**NamedBarrier** (coarse role-phase sync) [VERIFIED, FA4 `named_barrier.py`, issue #2418]:
- **IDs 0–15** (16 total). **ID 0 is reserved for `cute.arch.sync_threads()`** — start your enum at 1.
- DSL ops: `cute.arch.barrier_arrive(barrier_id, number_of_threads)` (arrive only), `cute.arch.barrier(barrier_id, number_of_threads)` (arrive+wait). No standalone wait.
- A producer↔consumer NamedBarrier counts **all** participating threads on both sides (e.g. 64 for two warps), unlike an mbarrier which counts only one side.
- FA4 builds an `IntEnum` of barrier IDs and an indexed subclass [`named_barrier.py:6`, `pipeline.py:166`]:
```python
class NamedBarrierFwdSm100(enum.IntEnum):
    Epilogue = enum.auto()    # = 1; barrier 0 reserved for sync_threads()
    TmemPtr  = enum.auto()
# indexed family: barrier_id + index
def arrive_w_index(self, index): cute.arch.barrier_arrive(self.barrier_id + index, self.num_threads)
```

### 6.7 Deadlock / heisenbug field guide

| Symptom | Cause | Fix |
|---|---|---|
| Manual-filled UMMA operand hangs at first MMA | used `PipelineTmaUmma`; its `producer_commit` is `pass` → `full_barrier` never armed | use `PipelineAsyncUmma` (real arrive) or hand-roll (§6.5) |
| Hang on a count-1 barrier from the MMA warp | all 32 lanes arrived → over-arrive → phase desync | wrap `tcgen05.commit` in `if lane_idx()==0` / `elect_one()` |
| Producer never wakes | stages not primed | `mbarrier_arrive(empty+s)` for every stage at init |
| Deadlock with arrive+wait on one barrier | self-arrive flips phase, own wait waits on next phase | split into full/empty barriers |
| `producer_group` count ≠ actual arrivers | `CooperativeGroup` size wrong | size = exact #threads that call `producer_commit` |
| **Passes WITH a `cute.printf`, hangs WITHOUT it** (the n≥128 heisenbug) | per-loop mbarrier **re-init** races the ring's async phase (the print's delay masks it); not a gmem RAW — gmem fences don't fix it | **init mbarriers ONCE**, use running-parity phases (`phase=(kt//stages)%2`), **never re-init across nested loops** [project `cute_qr_m6.py`, memory #14] |
| MMA reads stale operand despite a correct arrive | missing `fence_view_async_shared()` between fill and commit | add the async-shared fence before `producer_commit` |

**`defer_sync` note:** `create(..., defer_sync=True)` skips the pipeline's own cluster-init barrier — only safe if some other barrier (e.g. your one-time init `cute.arch.barrier()`) already synchronized all participants before first use; otherwise the ring starts on garbage phases.

**Caveats to verify on B200:** (a) the Python docstring mislabels `PipelineAsyncUmma.producer_commit` as the TMA no-op — confirm empirically (a 2-stage handoff that hangs under `PipelineTmaUmma` but completes under `PipelineAsyncUmma` is the test; the project already ran it); (b) the exact DSL spelling of the `tcgen05.commit` arrive wasn't found verbatim in the public docs — confirm via `dir(cutlass.cute.nvgpu.tcgen05)` / the wheel; (c) `>2`-stage phase bookkeeping across nested loops is the #1 heisenbug source — prefer `num_stages=2` (binary XOR) until the engine is correct, then widen.

---

## 7. Warp specialization, clusters & occupancy

> **Context for this project:** these are the levers behind the persistent cross-matrix M6 engine. The crown-jewel facts (manual roles + asymmetric `setmaxregister`, the `_override_create` frozen-dataclass extension, `tcgen05.commit` for UMMA-completion, the 512-col TMEM → 1-CTA/SM wall) live here. FA4 file refs are relative to `flash-attention-main 2/flash_attn/cute/`.

### 7.1 NO `warp_specialize` in 4.5.2 → manual `warp_idx` role branches

`warp_specialize` is a **Gluon-ism — it does NOT exist in cute-DSL 4.5.2.** The production form (confirmed by FA4) is a plain `if warp_idx == ...` branch that dispatches each warp to a role function, each branch immediately resizing its register budget.

**Rules:**
- `warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())` **before any warp-role branch** — uniformizes the value across the warp; without it the codegen/sync is wrong (plausible SIGSEGV/hang). *(verify on B200 that omitting it actually faults; the failure mode is the empirical claim.)*
- Branch on a **compile-time-known** warp-id range; the role functions are inlined per branch.
- Issue `cute.gemm` (tcgen05.mma) **warp-uniform, NOT inside `elect_one()`** (the compiler emits the single-thread issue); but `tcgen05.commit` **MUST** be `elect_one`-wrapped (else 32× redundant commits).

FA4 sm100, manual roles + per-branch reg resize (`flash_fwd_sm100.py:852, 1156, 1184, 1244`):
```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
    cute.arch.setmaxregister_decrease(self.num_regs_other)   # producer gives up regs
    self.load(...)
if warp_idx == self.mma_warp_id:
    cute.arch.setmaxregister_decrease(self.num_regs_other)
    self.mma(...)
if warp_idx <= self.softmax1_warp_ids[-1]:
    cute.arch.setmaxregister_increase(self.num_regs_softmax) # consumer grabs regs
    self.softmax_loop(...)
```

FA4 sm90, the minimal producer/consumer split (`flash_fwd_sm90.py:580-604`):
```python
if warp_idx < 4:                                   # Producer (TMA) warpgroup
    cute.arch.setmaxregister_decrease(self.num_producer_regs)
    self.load(...)
else:                                               # Consumer (MMA) warps
    cute.arch.setmaxregister_increase(self.num_mma_regs)
    self.mma(...)
```

### 7.2 Asymmetric register budgets: `setmaxregister` / `warpgroup_reg_*`

Two spellings of the SAME hardware `setmaxregister` mechanism — pick by granularity.

| API | Granularity | Notes |
|---|---|---|
| `cute.arch.setmaxregister_decrease(n)` | **per-warp** | producer/MMA warps shed regs |
| `cute.arch.setmaxregister_increase(n)` | **per-warp** | heavy-compute warps claim freed regs |
| `cute.arch.warpgroup_reg_dealloc(n)` | per-**warpgroup** (4 warps) | `@deprecated("use setmaxregister_decrease")` but still works |
| `cute.arch.warpgroup_reg_alloc(n)` | per-warpgroup | the warpgroup spelling, used in the 2-CTA FA4 kernel |

**Hard rules:**
- `n` must be a **multiple of 8**.
- It is **per-warp, NOT per-thread**, and it is a **HINT** — over-budget silently spills to local memory. **Validate `n_regs`/`n_spills==0` via SASS** (`fn.dump_to_object(path)` then `cuobjdump -res-usage`; ncu/nsys are gVisor-dead on Modal, so the SASS dump + timing-ablation are the only profiling tools).
- Order: `increase` after the low-reg warps `decrease`, in **ascending warp_idx order**, so the freed budget exists before a warp claims it. Total allocation across warps must fit the **64K-regs/SM** pool (≈ 512 regs/warp in a 4-warp warpgroup quad).
- FA4 budget tables by warpgroup count (`flash_fwd_sm90.py:215`): `{1:(256,56), 2:(240,24), 3:(160,32)}` = `(consumer_regs, producer_regs)`.

FA4 example budgets for a 5-role slice (panel 192 / mma 128 / load 40 / epi 48 / correction 88); minimal `192+128+40+48 = 408 ≤ 512` ✓ per warpgroup quad. The qr_v2 M2a win ("warp-split + setmaxregister that defeats the Gluon ib=16 wall") is exactly this pattern.

2-CTA / warpgroup spelling (`sm100_hd256_2cta_fmha_forward.py:808, 1285`):
```python
if warp_idx == self.load_warp_id:
    cute.arch.warpgroup_reg_dealloc(self.num_regs_other)    # producer releases
if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
    cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)    # softmax claims
```

### 7.3 NamedBarrier — cross-warp-role coordination

NamedBarriers coordinate role-warps that an mbarrier pair can't (e.g. softmax → correction stats handoff). **IDs 0–15 (16 total); ID 0 is reserved for `sync_threads()`** — start your enum at 1.

- Arrive-only: `cute.arch.barrier_arrive(barrier_id, number_of_threads)`.
- Arrive **+ wait**: `cute.arch.barrier(barrier_id, number_of_threads)` (no standalone wait exists).
- A producer↔consumer NamedBarrier counts **ALL** participating threads on both sides (e.g. 64 for two warps), unlike an mbarrier which counts one side only — get `number_of_threads` right or it hangs.
- A full `NamedBarrier` *class* is slated for a future release; in 4.5.2 use the `cute.arch.barrier*` ops (FA4 wraps them in its own subclass).

FA4 enum + indexed-barrier subclass (`named_barrier.py:6`, `pipeline.py:166`):
```python
class NamedBarrierFwdSm100(enum.IntEnum):
    Epilogue = enum.auto()       # = 1; barrier 0 reserved for sync_threads()
    TmemPtr  = enum.auto()
    SoftmaxStatsW0 = enum.auto()

# indexed family: barrier_id + index gives a per-stage/per-warp barrier
def arrive_w_index(self, index, ...):
    cute.arch.barrier_arrive(barrier_id=self.barrier_id + index, number_of_threads=self.num_threads)
def arrive_and_wait_w_index(self, index, ...):
    cute.arch.barrier(barrier_id=self.barrier_id + index, number_of_threads=self.num_threads)
```
Build one with `pipeline.NamedBarrier(barrier_id=int(NamedBarrierFwdSm100.TmemPtr), num_threads=...)` (`flash_fwd_sm100.py:875`); usage `sm_stats_barrier.arrive_w_index(index=stage*4 + warp_idx)` / `...arrive_and_wait_w_index(...)` (`flash_fwd_sm100.py:2319, 2473`).

**Single-elected-thread arrive** (per-warp count-1 signaling — avoid redundant arrives) (`flash_fwd_sm100.py:2355`):
```python
cute.arch.sync_warp()
with cute.arch.elect_one():
    pipeline_p_lastsplit.producer_commit_w_index(stage)
```
> ⚠️ Same elect-one rule as the async ring (§6): all-32 lanes over-arriving a count-1 barrier → phase desync → deadlock. Elect-one for any count-1 arrive.

### 7.4 Clusters & 2-SM (CtaGroup.TWO / TMEM two_ctas / cluster sync)

The 2-SM path threads together **three** flags that must agree:

| Knob | Set to | Where |
|---|---|---|
| MMA op | `tcgen05.CtaGroup.TWO` | `make_trivial_tiled_mma(..., cta_group=CtaGroup.TWO, ...)` |
| TMEM allocator | `is_two_cta=True` + `two_cta_tmem_dealloc_mbar_ptr=...` | `TmemAllocator(..., is_two_cta=True)` |
| TMA atom | a cluster-multicast op (`CopyBulkTensorTileG2SMulticastOp`) | `make_tiled_tma_atom_A/B(..., cluster_shape)` |
| launch | `cluster=[1, cluster_n, 1]` | `.launch(grid=..., block=..., cluster=...)` |

- `cluster=` at launch **must match** the cluster the tiled_mma/TMA atoms were built for; **omit for 1-CTA**; mismatch = illegal launch.
- 2-SM instruction shapes: `CtaGroup.TWO` → Mma-M ∈ {128, 256}; the C++ atom is `SM100_MMA_*_2x1SM_SS<...,256,256,...>`.
- **`TensorMemoryLayout(two_ctas)`** is the C++-side concept; in the Python DSL the 2-SM TMEM wiring is driven by the single `TmemAllocator(is_two_cta=...)` flag + `CtaGroup.TWO` — *(I did NOT find a Python `TensorMemoryLayout(two_ctas)` entry point; verify on B200 — the allocator flag is the confirmed handle.)*

**Cluster sync primitives** (`cute.arch`): `cluster_arrive_relaxed()` / `cluster_arrive(aligned=None)` then `cluster_wait()`; cluster dims via `cluster_dim()` / `cluster_idx()` / `block_idx_in_cluster()` / `cluster_size()`. For cross-CTA-in-cluster mbarrier arrives use the `peer_cta_rank_in_cluster=` arg of `mbarrier_arrive(...)`.

FA4 cluster-aware pipeline create (`sm100_hd256_2cta_fmha_forward.py:607`) passes `cta_layout_vmnk=cluster_layout_vmnk` and `defer_sync=True` (defers the cluster init barrier); the consumer group size includes the cluster factor (`len(...) * threads_per_warp * cluster_shape_mnk[0]`, `:635`).

### 7.5 Persistent tile scheduler + occupancy

`StaticPersistentTileScheduler` (`cutlass.utils`) launches **min(#SMs, #work-tiles)** CTAs; each CTA grid-strides over many tiles. This is the drop-in skeleton for a **per-matrix** persistent qr_v2 engine (one CTA factors a whole matrix, grabs the next from the queue).

**Driver loop** (`flash_fwd_sm90.py:1001, 1046, 1266`):
```python
tile_scheduler = TileSchedulerCls()
work_tile = tile_scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:          # the ONLY loop exit — early exit hangs peers
    ...                                 # process work_tile.tile_idx
    tile_scheduler.advance_to_next_work()
    work_tile = tile_scheduler.get_current_work()
```

**Grid sizing + grid-stride advance** (`tile_scheduler.py:337, 364`):
```python
sm_count = hardware_info.get_device_multiprocessor_count()           # B200 ≈ 148
grid_x = cutlass.min(sm_count, params.total_blocks_cluster * cluster_m)
...
def advance_to_next_work(self):
    self._tile_idx += cute.arch.grid_dim()[0]    # grid-stride (+= grid), NOT +1
    return self.get_current_work()
```

**Rules:**
- `is_valid_tile` is the **only** loop exit; an early `break`/`return` in one CTA hangs its cluster peers (and dynamic control flow has no `break` anyway).
- Advance is **grid-stride** (`+= grid_dim()[0]`), not by 1.
- The scheduler object must survive the `while` loop's SSA region → it implements `__extract_mlir_values__` / `__new_from_mlir_values__` (the loop-carry protocol; `tile_scheduler.py:374`). Any custom Python object carried through a dynamic loop needs this pair.
- Linear tile-id → coords via fast `divmod` against precomputed divisors (`get_current_work`, `tile_scheduler.py:350`).
- **Grep-clean alternative:** if scheduler class names leak `stream`/`graph` (⚠️ **avoid the `Clc*` cluster-launch-control classes — their docstrings can leak the banned substrings**), hand-roll the work queue with the atomic cross-CTA barrier (`barrier.py` recipe: `dsl_user_op` + `red.release`/`ld.acquire` inline-PTX on a gmem semaphore — `wait_eq`/`arrive_inc`). It is grid-sync without cooperative launch and grep-clean.

**Occupancy oracle — `HardwareInfo`** (`cutlass.utils`):
```python
hw = utils.HardwareInfo(device_id=0)
hw.get_device_multiprocessor_count()       # SM count (B200 ≈ 148)
hw.get_max_active_clusters(cluster_size)    # compiles a probe → cuOccupancyMaxActiveClusters
hw.get_l2_cache_size_in_bytes()
```
`get_max_active_clusters` requires an initialized CUDA driver/context (it JITs+launches a probe). Use it to size a persistent grid rather than hardcoding.

**What gates CTAs/SM (the occupancy wall):**
- **TMEM:** 512 columns/SM total. A full-width F32 accumulator (512 cols) monopolizes TMEM → **1 CTA/SM**. To get 2–3 CTAs/SM, shrink the accumulator (`tmem.allocate(256)` → ~2 CTAs/SM) or share via `TmemBufferPool`. *This is the structural wall behind the falsified per-matrix tcgen05 engine (M3c = 1 CTA/SM → ~1.0–1.2× V10; the 0.87× breakthrough needs the M6 cross-matrix masking).*
- **Registers:** the 64K-regs/SM pool — asymmetric `setmaxregister` is how you keep a heavy role under budget so more CTAs co-reside.
- **SMEM:** 228 KB/SM (verify on B200) — size your ring stages against it; overflow = launch failure.

### 7.6 Extending frozen CUTLASS dataclasses — the FA4 `_override_create` pattern

CUTLASS pipeline/scheduler classes are `@dataclass(frozen=True)`, so you can't subclass-and-init to add methods (e.g. an `extra_tx_count` `producer_acquire`, or an `elect_one`-gated commit). FA4's keystone trick: build the parent via its own `.create`, then **rebind `__class__`** to a child subclass via `object.__setattr__` (bypasses the frozen guard).

`pipeline.py:20-30` — the generic factory:
```python
def _override_create(parent_cls, child_cls):
    @staticmethod
    def create(*args, **kwargs):
        obj = parent_cls.create(*args, **kwargs)
        # can't assign __class__ directly — the dataclass is frozen
        object.__setattr__(obj, "__class__", child_cls)
        return obj
    return create
```
Install it after defining the child (`pipeline.py:330`):
```python
PipelineTmaAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaAsync)
```
The child then adds methods the frozen parent lacked — e.g. the `extra_tx_count` producer_acquire (`pipeline.py:304`) and the `elect_one_commit` knob (`pipeline.py:101`). A custom Python object carried through a dynamic loop (pipeline state, scheduler) must also implement `__extract_mlir_values__` / `__new_from_mlir_values__` (`pipeline.py:78`) so it survives the MLIR SSA region — the same loop-carry protocol the scheduler uses (§7.5).

### 7.7 Quick gotcha list

- `make_warp_uniform` **before** every warp-role branch (verify the omission-faults claim on B200).
- `setmaxregister` is per-warp, multiple-of-8, a HINT — confirm `spills==0` in SASS.
- `cute.gemm` warp-uniform (no `elect_one`); `tcgen05.commit` **needs** `elect_one`; any count-1 NamedBarrier/mbarrier arrive needs `elect_one` (over-arrive → phase-desync deadlock).
- NamedBarrier IDs 0–15; ID 0 reserved; count **both** sides' threads.
- `cluster=` at launch must match the atoms' cluster (omit for 1-CTA); 2-SM = `CtaGroup.TWO` + `is_two_cta=True` + multicast TMA atom, all three together.
- Persistent loop: `is_valid_tile` is the only exit; advance is `+= grid_dim()[0]`; carry the scheduler via the MLIR-values protocol; avoid `Clc*` classes (`stream`/`graph` docstring leak).
- TMEM 512 cols = the occupancy wall: full-width acc → 1 CTA/SM.

---

Both references check out. The `producer_commit`-is-noop docstring artifact is confirmed present in FA4's `pipeline.py` (it re-classes all four pipeline variants via `_override_create`). Here is the section.

## 8. Gotchas & version traps (4.5.2-specific)

> Every trap below cost at least one B200/Modal run. Format: **TRAP → what fails → the right form.** Pinned to `nvidia-cutlass-dsl==4.5.2`, CUDA 12.9, sm_100a. When in doubt, `dir()`/`inspect.signature` the installed wheel — the public docs auto-truncate and several API pages 404.

### 8.1 The killer four (silent-wrong / silent-hang)

| # | TRAP | What fails | The right form |
|---|---|---|---|
| 1 | **`cutlass.and_` / `cute.core.and_` for a bitmask** | They are **logical** and → produce **NaN** on float operands; tf32x3 rel error stuck at ~8.8e-4 (= plain 1×tf32, i.e. the split silently did nothing) | Python bitwise **`&`** on an `Int32` bitcast view |
| 2 | **`cute.bitcast(x, dtype)`** | **Does not exist** in 4.5.2 (build-guide L439 is wrong) — `AttributeError` / trace failure | `.bitcast` is a **value method**: `x.bitcast(cutlass.Int32)` |
| 3 | **`cute.arch.sqrt(x)`** | Not present under `cute.arch` → `AttributeError` | `cute.math.sqrt(x, fastmath=True)` (likewise `cute.math.rsqrt/exp2`, **not** `cute.arch.sqrt`; note `cute.arch.{exp2,rcp_approx,fmax}` *do* exist as the low-level scalar forms) |
| 4 | **`warp_specialize(...)`** | It's a **Gluon-ism** — absent in 4.5.2 cute-DSL | Manual warp roles: `if warp_idx == ...:` branch (after `make_warp_uniform`) + `cutlass.pipeline` classes + `setmaxregister_*` |

### 8.2 The tf32x3 in-register split — the only form that survives the optimizer

`-8192 = 0xFFFFE000` clears the low 13 mantissa bits, giving the TF32 "hi" limb; `lo = x - hi` is the residual. **Validated EXACT** (maxabs 0.0; GEMM-1 rel 8.25e-7) in `experiments/cute_gemm_tf32x3.py`.

```python
xi = x.bitcast(cutlass.Int32)
hi = (xi & cutlass.Int32(-8192)).bitcast(cutlass.Float32)   # 0xFFFFE000 mask
lo = x - hi
```

**TRAP → both "obvious" alternatives FOLD TO IDENTITY under the optimizer:**
- The arithmetic **Dekker split** `c = K*x; hi = c - (c - x)` → collapses to `hi = x` → rel stuck at 8.8e-4 (1×tf32, no win).
- The **TFloat32 round-trip** `x.to(cutlass.TFloat32).to(cutlass.Float32)` → same collapse, same stuck rel.

→ Only the **bit-AND** split is not constant-folded away. There is **no `tf32x3` MMA op**; you emulate it as 3× `MmaTF32Op` (hi·hi, hi·lo, lo·hi) into one **FP32** accumulator. `MmaTF32Op` forces inst-K **= 8** (raises otherwise). (Source: `cutedsl-m3c-validated-primitives` memory; CUTLASS 3xTF32 technique [discussions/361](https://github.com/NVIDIA/cutlass/discussions/361).)

### 8.3 Pipeline / mbarrier deadlock traps (Blackwell UMMA)

**TRAP → `PipelineTmaUmma`/`PipelineAsyncUmma` with a MANUAL `st.shared`-filled operand → MMA consumer hangs forever.**
Root cause (definitively diagnosed via `experiments/cute_pipesrc.py`): the **`producer_commit` body is literally `pass`**. The Python-DSL docstring says *"TMA producer commit is a noop since the TMA instruction itself updates the transaction count"* — and FA4 re-classes **all four** pipeline variants over this same parent (`_override_create`, `pipeline.py:330/380/391/402`), so the misleading docstring rides along. For a real TMA producer the `full_barrier` is armed by the TMA transaction-count, so the noop is correct; but a hand `st.shared` fill **never arrives the barrier** → the UMMA consumer's `consumer_wait` blocks forever.

→ **Right forms (two options):**
1. **Use `PipelineAsyncUmma`** (async-thread producer → UMMA consumer): per the C++ symmetry rule its `producer_commit` is a **real `mbarrier.arrive`**. Fill smem → `cute.arch.fence_view_async_shared()` (or `fence_proxy("async.shared", space="cta")`) → `producer_commit`; MMA warp `consumer_wait` → `cute.gemm` → `consumer_release(..., cta_group=...)`. *(The Python docstring still mislabels this `producer_commit` as the TMA noop — verify it emits a real arrive on B200.)*
2. **Hand-roll low-level mbarriers** (the validated 1.58× ring, `experiments/cute_ring_g1.py`) — see 8.4.

⚠️ **Doc-vs-source conflict:** trust the **C++ symmetry rule** + sm100 source over the Python docstrings — the noop text was copy-pasted onto the UMMA/async classes where it does not apply.

### 8.4 elect-one / arrive-count traps

**TRAP → `tcgen05.commit(mbar)` issued by all 32 lanes → over-arrives a count-1 barrier → phase desync → deadlock.**
→ Must be **elect-one**:
```python
if cute.arch.lane_idx() == 0:
    tcgen05.commit(empty_mbar + s)      # UMMA-completion → frees the stage
```
(`tcgen05.commit(mbar, mask=None, cta_group=ONE)` is the load-bearing primitive — UMMA arrives on `mbar` when the mma-group completes; it's what `PipelineUmmaAsync.producer_commit` calls internally.) **Conversely:** `cute.gemm` / `tcgen05.mma` and `cute.copy(TMA...)` **self-elect** — do **NOT** wrap them in `elect_one()`, that deadlocks (cheatsheet L226-230). Rule of thumb: **`cute.gemm` = warp-uniform, no elect; `tcgen05.commit` = needs elect-one.**

**TRAP → producer `mbarrier_init` count ≠ actual arriving thread count** → barrier never completes or completes early (phase skew). Set `NPROD` = exact number of producer threads that will `mbarrier_arrive` per stage (the validated ring uses 96 = 3 warps).

**TRAP → arrive and wait on the SAME barrier** → the arrive that satisfies the count flips the phase, so your own wait then waits on the *next* phase → deadlock. → Keep the **full/empty split** (two mbarriers per stage), exactly as the pipeline classes do. (CUTLASS issues [#2404](https://github.com/NVIDIA/cutlass/issues/2404), [#2418](https://github.com/NVIDIA/cutlass/issues/2418).)

### 8.5 The validated hand-rolled ring (the crown-jewel pattern, `cute_ring_g1.py`)

Init **once** (tidx==0): `mbarrier_init(full+s, NPROD)`, `mbarrier_init(empty+s, 1)`, **prime** `mbarrier_arrive(empty+s)`, `mbarrier_init_fence()`, one `cute.arch.barrier()`. Producer (warps 0-2): `mbarrier_wait(empty+s, ph)` → fill → `fence_view_async_shared()` → `mbarrier_arrive(full+s)`, with `s=kt%2, ph=(kt//2)%2`. Consumer (warp 3): `mbarrier_wait(full+s, ph)` → 3-pass `cute.gemm` ACC-accumulate → elect-one `tcgen05.commit(empty+s)`. **Measured 1.58× faster** than the per-tile-barrier K-loop, both correct.

**TRAP → re-init mbarriers inside a nested per-column loop → heisenbug** (`cute_qr_m6.py`: hangs without device prints, *passes* with a `cute.printf` because the print delay masks the race; gmem fences don't fix it → it's a timing race, not a gmem RAW). → **Init mbarriers ONCE; use running-parity phases**, never re-init across nested loops.

### 8.6 Fences you cannot omit (silent corruption if skipped)

| Situation | Required fence |
|---|---|
| after `st.shared` fill, before signalling MMA / TMA-store consumer | `cute.arch.fence_view_async_shared()` (`flash_bwd_mla_sm100.py:1886`) |
| after a **TMEM store**, before signalling a pipeline | `cute.arch.fence_view_async_tmem_store()` (`flash_fwd_sm100.py:2353`) |
| after **TMEM→reg** read, before reusing those regs / releasing the stage | `cute.arch.fence_view_async_tmem_load()` (`sm100_hd256_2cta_fmha_forward.py:1557`) |
| generic async proxy, explicit space | `cute.arch.fence_proxy("async.shared", space="cta")` (`flash_bwd_mla_dq_dqv_sm100.py:837`) |

→ Also: `tmem.wait_for_alloc()` **must** precede `retrieve_ptr` in **every** warp that reads the accumulator (else a garbage TMEM base pointer → fault); and `make_warp_uniform(warp_idx)` is **required** before any warp-role branch.

### 8.7 Swizzled hand-fill — it WORKS (refutes "corrupt")

**TRAP → the belief that writing a swizzled MMA operand by logical coordinate silently corrupts it.** It does **not**. Validated on B200 (`cutedsl-m3c-validated-primitives` #2): with `make_smem_layout_a/b(tiled_mma,(128,256,32),Float32,stages)` → layout `((R,8),1,4,1):((32,1),0,8,0)`, a direct fill by logical coord is correct, and you can apply the unit-lower V mask **during** the fill:
```python
sX[(row, ki), 0, kb, 0] = val          # k = kb*8 + ki ; correct under the swizzle
```
The MMA smem descriptor only encodes the layout (LBO/SBO/swizzle) — it does **not** care how the bytes arrived (gau-nernst: "TMA is not strictly required"), so a manual fill matching the layout feeds `tcgen05.mma` fine. (Pair with the `PipelineAsyncUmma`/hand-mbarrier handshake from 8.3/8.4 to tell UMMA the fill is ready.)

### 8.8 Environment / import / tracing traps

- **TRAP → `import cutlass.cute.experimental`** → raises `NotImplementedError` below **CUDA 13.1**; the eval/Modal image is 12.9 → the module is **absent**. Don't use it. At 12.9, `barrier`/`barrier_arrive` lower to inline `bar.sync`/`bar.arrive` PTX.
- **TRAP → `debian_slim + torch-cu128 (2.11)` Modal image** → **SIGABRTs** (`LLVM ERROR: unsupported operation`) from a cuda-python-13 vs torch-2.11-cuda-bindings (<13) **ABI clash**. → Use **`torch==2.12.0`** (+ `nvidia-cutlass-dsl==4.5.2 cuda-python==13.0`, base `nvidia/cuda:12.9.1-devel`, py3.13).
- **TRAP → nested `def` capturing a pipeline object inside dynamic control flow** → *"closures not supported in dynamic control flow."* → **INLINE** the mma/readback bodies; no inner functions over pipeline objs.
- **TRAP → omitting the stage index in a staged fragment slice** → wrong stage read. → It's mandatory: `frag[(None, None, kb, 0)]`.
- **TRAP → passing a dynamic IR value to `range_constexpr`/native Python control flow** (or `break`/`continue`/type-change inside a dynamic block) → trace failure. → `cutlass.range(...)` for dynamic loops, `range_constexpr(...)` only with Constexpr bounds; `cutlass.const_expr(...)` gates compile-time branches.
- **TRAP → scalar `t[i]` indexing assumed forbidden** → it actually **works** in a `@cute.jit` dynamic loop (`Hm[r,c]`), but is **slow** — use it only for correctness floors; prefer the partitioned/TV-layout + `cute.copy` path for perf. (Reconciles the cheatsheet's "NOT scalar `t[i]`" with the validated `Hm[r,c]`.)
- **TRAP → `print(...)` in a kernel** runs at **trace/JIT time**, not on device. → `cute.printf(...)` for device prints (and its indent must match the loop-body indent, or it shifts the very timing that masks the 8.5 heisenbug).

### 8.9 The `stream`/`graph` substring scan (auto-DQ)

**TRAP → any submission containing the substrings `stream` or `graph` — including in comments/docstrings — is rejected** by a naive static scan. Notably `cute.arch.sm_id()`'s **docstring literally contains "Streaming Multiprocessor"**, and `Clc*` (cluster-launch-control) class docstrings can leak `stream`/`graph`. → `cute.compile`/`.launch` work with **no stream arg** (don't copy the DG/`export_to_c` `stream=` plumbing); **avoid `Clc*` classes**; `grep -niE "stream|graph" submission.py` MUST be empty before every submit.

### 8.10 Profiling reality (so you don't chase a non-existent tool)

**TRAP → `ncu`/`nsys` on Modal** → **dead** (gVisor sandbox: `/dev/nvidia-caps` absent → `perfmon LibraryNotLoaded`; version-independent). → Timing-ablation is the only profiler. For register/spill validation (which `setmaxregister` correctness demands — confirm `n_spills=0`): the cute cubin is in-memory → `fn.dump_to_object(path)` then `cuobjdump -res-usage` (or `dump_kernel_attributes` via the CUDA driver, needs `--keep-cubin`). To swap a newer/`-O3` ptxas: `CUTE_DSL_PTXAS_PATH=... CUTE_DSL_KEEP_PTX=1`.

---

Files backing the validated claims (all absolute): `/Users/raymond/Downloads/SubPY/.claude/worktrees/vibrant-proskuriakova-9a15c8/experiments/cute_gemm_tf32x3.py` (8.2), `.../experiments/cute_ring_g1.py` (8.4/8.5), `.../experiments/cute_qr_m6.py` (8.5 heisenbug), `.../experiments/cute_pipesrc.py` (8.3 diagnosis), `.../experiments/cute_qr_m3c_tc.py` (8.7); FA4 `/Users/raymond/Downloads/flash-attention-main 2/flash_attn/cute/pipeline.py:330,380,391,402` (8.3 `_override_create` re-class confirmed present).

---

I'll write Section 9. Let me silently load the visualization context isn't needed here—this is a markdown reference section. Let me compose it directly from the research pool.

## 9. Recipes & quick API index

> All FA4 paths are relative to `flash_attn/cute/`. Line numbers are from the mined snapshot — treat as "look near here," not exact across versions. CUTLASS source paths are under `python/CuTeDSL/cutlass/`. Anything not nailed to a verbatim snippet is marked **(verify on B200)**.

### 9.1 "I want to X → use Y" recipe table

| I want to… | Use | Reference |
|---|---|---|
| **Load a GMEM tile via TMA** | Build atom on host: `cute.nvgpu.make_tiled_tma_atom_A/B(op, t, smem_layout, mma_tiler, tiled_mma, cluster_shape)` → `(atom, retiled_t)`; in kernel `cpasync.prefetch_descriptor(atom)` once, then `cpasync.tma_partition(...)` + `cute.copy(atom, gSrc, sDst, tma_bar_ptr=handle.barrier)` | atom: `sm100_hd256_2cta_fmha_forward.py:457`; prefetch: `flash_fwd_sm100.py:855`; partition: `flash_fwd_sm100.py:1415`; copy: `sm100_hd256_2cta_fmha_forward.py:947` |
| **Build the TMA store (S2G)** | `cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mO, sO_layout_2d, epi_tile)` | `flash_fwd_sm100.py:618` |
| **Software-pipeline a GEMM (TMA→UMMA)** | `pipeline.PipelineTmaUmma.create(num_stages, producer_group, consumer_group, tx_count=, barrier_storage=, cta_layout_vmnk=).make_participants()` → producer `acquire_and_advance()`/`tail()`, consumer `wait_and_advance()`/`release()` | create: `sm100_hd256_2cta_fmha_forward.py:607`; producer drive: `:947`; consumer drive: `:1088`; drain: `flash_fwd_sm100.py:1030` |
| **Compute `tx_count` per stage** | `cute.size_in_bytes(dtype, smem_layout)` | `sm100_hd256_2cta_fmha_forward.py:487` |
| **Feed a compute-produced operand to UMMA** (hand-filled / masked / transposed smem, NOT TMA) | `pipeline.PipelineAsyncUmma.create(...)` (no `tx_count`); producer warp: write smem → `cute.arch.fence_view_async_shared()` → `producer_commit` (a **real** `mbarrier.arrive`); MMA warp: `consumer_wait` → `cute.gemm` → `consumer_release(cta_group=)` | create: `sm100_hd256_2cta_fmha_forward.py:635`; producer fill+commit: `flash_bwd_mla_sm100.py:1885`; ⚠️ see §9.3 deadlock note |
| **— low-level fallback** (PipelineAsyncUmma fights you) | Hand-roll `full_mbar[S]`/`empty_mbar[S]`: producer `mbarrier_wait(empty,ph)`→fill→`fence_view_async_shared()`→`mbarrier_arrive(full)`; consumer `mbarrier_wait(full,ph)`→3×`cute.gemm`→**elect-one** `tcgen05.commit(empty)` | validated ring recipe, §9.4 below |
| **Warp-specialize producer/consumer** (no `warp_specialize` in 4.5.2) | `warp_idx = make_warp_uniform(cute.arch.warp_idx())`; `if warp_idx == ROLE:` branch; inside: `setmaxregister_decrease(n)` (producer/MMA) / `setmaxregister_increase(n)` (heavy compute); cross-role sync via `NamedBarrier` | dispatch: `flash_fwd_sm100.py:852`; sm90 form: `flash_fwd_sm90.py:580`; warpgroup form: `sm100_hd256_2cta_fmha_forward.py:808/1285` |
| **Cross-role named-barrier sync** | `IntEnum` of barrier ids starting at **1** (id 0 = `sync_threads`); `pipeline.NamedBarrier(barrier_id=, num_threads=)`; `.arrive()`/`.arrive_and_wait()` (+ FA4 `_w_index` variants for barrier families) | enum: `named_barrier.py:6`; build: `flash_fwd_sm100.py:875`; indexed: `pipeline.py:166` |
| **Allocate + read TMEM** | `utils.TmemAllocator(holding_buf, barrier_for_retrieve=, allocator_warp_id=, is_two_cta=)`; MMA warp: `allocate(get_max_tmem_alloc_cols("sm_100"))` → `wait_for_alloc()` → `retrieve_ptr(acc_dtype)`; consumer warps: `wait_for_alloc()` → `retrieve_ptr(...)` | setup: `flash_fwd_sm100.py:885`; alloc: `:1187`; consumer: `:1246` |
| **Read accumulator TMEM→registers (t2r)** | `cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(n)), acc_dtype)` → `tcgen05.make_tmem_copy(atom, tTMEM)` → `cute.copy(...)` → **`cute.arch.fence_view_async_tmem_load()`** before reuse/release | `sm100_hd256_2cta_fmha_forward.py:1547`; 1-CTA: `flash_fwd_sm100.py:2286` |
| **Write registers→TMEM (r2t)** | `St32x32bOp(Repetition(n))` atom → `make_tmem_copy` → `cute.copy(...)` → **`cute.arch.fence_view_async_tmem_store()`** before signalling | `sm100_hd256_2cta_fmha_forward.py:1586` |
| **Build a tcgen05 tiled MMA** | `sm100_utils.make_trivial_tiled_mma(a_dtype, a_major, b_major, acc_dtype, cta_group, mma_tiler_mn[, OperandSource.TMEM])` | `sm100_hd256_2cta_fmha_forward.py:402` |
| **Run MMA, accumulate over K** | First k-step `tiled_mma.set(tcgen05.Field.ACCUMULATE, False)` (overwrite), then `True`; `cute.gemm(tiled_mma, acc, A, B, acc)` — issue **warp-uniform, NOT in `elect_one`** | helper: `blackwell_helpers.py:96`; loop: `sm100_hd256_2cta_fmha_forward.py:1088` |
| **Do tf32x3** (no native op — emulate) | In-register split `hi = (x.bitcast(Int32) & Int32(-8192)).bitcast(Float32)`; `lo = x - hi`; feed hi/lo as 3 `MmaTF32Op` passes into one **F32** acc | split validated `cute_gemm_tf32x3` (rel 2.89e-6); §9.5 below + CUTLASS 3xTF32 discussion #361 |
| **Cross-CTA barrier** (grep-clean, no cooperative launch) | `@dsl_user_op` + `llvm.inline_asm`: `red.release.gpu.global.add.s32` (arrive) + `ld.global.acquire.gpu.b32` (spin); thread-0-only `wait_eq`/`arrive_inc` on a GMEM semaphore tensor | `barrier.py:8` (ld_acquire), `:39` (red_release), `:55` (wait_eq/arrive_inc); usage `flash_bwd_sm90.py:1734` |
| **Persistent per-matrix scheduler** | `grid = min(#SMs, #tiles)` via `HardwareInfo().get_device_multiprocessor_count()`; `while work.is_valid_tile:` … `advance_to_next_work()` (`+= grid_dim()[0]`) `get_current_work()` | grid: `tile_scheduler.py:337`; advance: `:364`; loop: `flash_fwd_sm90.py:1001` |
| **Swap in a newer/-O3 ptxas** | env `CUTE_DSL_PTXAS_PATH` + `CUTE_DSL_KEEP_PTX=1` (+ `CUTE_DSL_DUMP_DIR`); `patch()` **before** `import cutlass` | `cute_dsl_ptxas.py:59/132` |
| **Dump regs/spills (ncu/nsys are gVisor-dead on Modal)** | `fn.dump_to_object(path)` → `cuobjdump -res-usage`; or `dump_kernel_attributes` (`NUM_REGS`/`LOCAL_SIZE_BYTES`, needs `--keep-cubin`) | `cute_dsl_utils.py:127`; project memory |

### 9.2 Mandatory async-fence cheat (correctness, not perf)

| After you… | …before you… | Fence |
|---|---|---|
| write smem (manual / cp.async) | signal a pipeline an async/UMMA consumer reads | `cute.arch.fence_view_async_shared()` — `flash_bwd_mla_sm100.py:1886` |
| write TMEM (r2t) | signal MMA/another warp | `cute.arch.fence_view_async_tmem_store()` — `flash_fwd_sm100.py:2353` |
| read TMEM→regs (t2r) | reuse those regs / release the stage | `cute.arch.fence_view_async_tmem_load()` — `sm100_hd256_2cta_fmha_forward.py:1557` |
| generic async-proxy | pass smem to TMA-store/UMMA | `cute.arch.fence_proxy("async.shared", space="cta")` — `flash_bwd_mla_dq_dqv_sm100.py:837` |

### 9.3 PipelineAsyncUmma deadlock warning (cost the project multiple days)

The Python-DSL docstrings for `PipelineUmmaAsync`/`PipelineAsyncUmma` **wrongly** say `producer_commit` is "a noop since the TMA instruction updates the transaction count" — that text is copy-pasted from `PipelineTmaAsync`. The C++ symmetry rule is authoritative: `producer_commit` is a **real `mbarrier.arrive`** when the producer is an *async thread* (your `st.shared` fillers), and a no-op only when a TMA/UMMA hardware op arms the barrier. **Trap:** the high-level `PipelineTmaUmma.producer_commit` body is literally `pass` — so if you drive a **manual `st.shared` fill** through a TMA-flavored pipeline, the full-barrier is never armed and the MMA consumer waits forever. Use `PipelineAsyncUmma` (real arrive) for manual fills, or hand-roll mbarriers (§9.4). **(verify on B200: confirm `PipelineAsyncUmma.producer_commit` emits a real arrive via PTX before relying on it.)**

### 9.4 Hand-rolled mbarrier async-ring (validated, 1.58× over per-tile-barrier K-loop)

The crown-jewel deployable pattern when canned pipelines don't fit a hand-filled UMMA operand:

```python
# SharedStorage: full_mbar[stages], empty_mbar[stages], acc_done[1]  (all MemRange[Int64])
# init (tidx==0 only, ONE-TIME):
#   mbarrier_init(full+s, NPROD); mbarrier_init(empty+s, 1); mbarrier_init(accd, 1)
#   mbarrier_arrive(empty+s)                       # PRIME the empty barriers
#   mbarrier_init_fence(); cute.arch.barrier()     # single one-time CTA barrier
# PRODUCER warps (s=kt%2, ph=(kt//2)%2):
#   mbarrier_wait(empty+s, ph); fill smem[...,s]
#   cute.arch.fence_view_async_shared(); mbarrier_arrive(full+s)
# CONSUMER warp:
#   mbarrier_wait(full+s, ph); 3-pass cute.gemm ACC-accumulate into TMEM
#   if lane_idx()==0: tcgen05.commit(empty+s)      # UMMA-completion -> frees stage
# after loop:  if lane0: tcgen05.commit(accd); then ALL threads mbarrier_wait(accd,0); readback
```

- **`tcgen05.commit(mbar, mask=None, cta_group=ONE)`** is the missing primitive: UMMA arrives on `mbar` when the mma-group completes (what `PipelineUmmaAsync.producer_commit` calls internally). Wrap it in **elect-one** — all-32 lanes over-arriving a count-1 barrier desyncs the phase → deadlock.
- **`num_stages=2`** → phase is binary XOR; S0/S1 are different TMEM offsets, not one slot with a flip bit.
- **>2 stages / nested loops:** init mbarriers **once** with running-parity phases; **do NOT re-init across nested loops** — that races the ring's async phase (the `cute_qr_m6.py` heisenbug: hangs without prints, passes with `cute.printf` = a timing race).
- Arrive an mbarrier and wait on it on **different** barriers (full/empty split) — a self-arrive flips the phase and your own wait then blocks on the next phase.

### 9.5 tf32x3 in-register split — the working recipe + 3 dead alternatives

```python
xi = x.bitcast(cutlass.Int32)
hi = (xi & cutlass.Int32(-8192)).bitcast(cutlass.Float32)   # -8192 = 0xFFFFE000, clears low 13 mantissa bits
lo = x - hi
# then 3 MmaTF32Op passes (hi·hi, hi·lo, lo·hi) into ONE Float32 acc
```
Validated EXACT (maxabs 0.0 on the split; GEMM rel 8.25e-7 / 2.89e-6). **Three traps that all silently collapse to 1×tf32 (rel stuck ~8.8e-4):**
- `cute.bitcast(...)` **does not exist** — bitcast is a value **method** `x.bitcast(dtype)`.
- `cutlass.and_` / `cute.core.and_` are **logical** AND (produce NaN) — use Python `&` (bitwise).
- The Dekker split `c-(c-x)` **and** the `x.to(TFloat32).to(Float32)` round-trip both get optimized away. Only the bit-AND survives.

### 9.6 Compact API quick-index (symbol → meaning → ref)

**Decorators / launch / compile**
| Symbol | Meaning | Ref |
|---|---|---|
| `@cute.jit` | host JIT fn (sets up layouts, launches kernels) | [dsl_introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) |
| `@cute.kernel` | GPU kernel; launchable only from a `@cute.jit` | dsl_introduction |
| `kernel(args).launch(grid=[...], block=[...], cluster=..., smem=...)` | enqueue; 3-elt seqs; `cluster` must match atom build | `elementwise_add.py` |
| `cute.compile(host_fn, *example_args, options="--generate-line-info")` | precompile → stable callable; bypasses cache | `elementwise_add.py`; [aot doc](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html) |
| `fn.dump_to_object(path)` | dump in-memory cubin (then `cuobjdump -res-usage`) | project memory |

**Types / control flow**
| Symbol | Meaning | Ref |
|---|---|---|
| `cutlass.Constexpr` / `Constexpr[T]` | compile-time baked value | [types](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html) |
| `cutlass.const_expr(cond)` | compile-time branch (untaken path not emitted) | `rmsnorm.py` |
| `cutlass.range(n[, unroll=, prefetch_stages=])` | runtime IR loop (+ SW-pipeline) | [control_flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html) |
| `cutlass.range_constexpr(n)` | fully unrolled at trace time (bounds must be Constexpr) | `rmsnorm.py` |
| `value.to(dtype)` / `value.bitcast(dtype)` | numeric cast / bit reinterpret (method, **not** `cute.bitcast`) | §9.5 |
| (gotcha) | dynamic blocks: no `break`/`continue`/type-change; **no nested `def` closing over pipeline objs** — inline the body | project memory #5 |

**Runtime / tensors**
| Symbol | Meaning | Ref |
|---|---|---|
| `from_dlpack(t, assumed_align=16)` | torch→cute bridge (only one) | `call_bypass_dlpack.py` |
| `.mark_layout_dynamic()` / `.mark_compact_shape_dynamic()` | runtime-vary shape/strides | cheatsheet §2 |
| `cute.make_tensor(ptr, layout)` / `make_ptr` | pointer-bypass tensor build | `call_bypass_dlpack.py` |
| (gotcha) | scalar `Hm[r,c]` works in dynamic loops but is **slow** — TV/partition path for perf | project memory |

**arch intrinsics**
| Symbol | Meaning | Ref |
|---|---|---|
| `cute.arch.thread_idx()/block_idx()/grid_dim()` | 3-tuple of Int32 | [cute_arch](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html) |
| `cute.arch.warp_idx()/lane_idx()` | warp/lane id | cute_arch |
| `make_warp_uniform(x)` | uniformize before warp-role branch (**required**) | `flash_fwd_sm100.py:852` |
| `with cute.arch.elect_one():` | one-thread region (do NOT wrap `cute.copy`/`cute.gemm`) | `flash_fwd_sm100.py:2355` |
| `barrier()/barrier_arrive(barrier_id=, number_of_threads=)` | named-barrier arrive+wait / arrive | issue #2418; `pipeline.py:166` |
| `sync_threads()/sync_warp()` | `__syncthreads` / warp sync | cheatsheet §3 |
| `mbarrier_init/arrive/wait/try_wait(mbar, [phase])` | low-level mbarrier | `dense_gemm.py` |
| `mbarrier_arrive_and_expect_tx(mbar, bytes)` / `mbarrier_expect_tx` | arm transaction-byte barrier (TMA) | cute_arch |
| `cp_async_commit_group()/cp_async_wait_group(n)` | non-bulk cp.async ordering | cheatsheet |
| `setmaxregister_increase/decrease(n)` | per-warp reg budget (multiple of 8; fits 64K/SM) | `flash_fwd_sm90.py:581` |
| `warpgroup_reg_alloc/dealloc(n)` | warpgroup-granular reg realloc (`@deprecated` alias) | `sm100_hd256_2cta_fmha_forward.py:785` |

**math**
| Symbol | Meaning | Ref |
|---|---|---|
| `cute.math.sqrt(x, fastmath=True)` / `rsqrt` / `exp2` | elementwise math (use **`exp2`** for softmax; **not** `cute.arch.sqrt`) | `rmsnorm.py`, `softmax.py:140` |
| `cute.arch.rcp_approx(a)` / `fmax(a,b)` / `exp2(a)` | per-scalar NVVM intrinsics | cute_arch |
| `cute.arch.shuffle_sync_bfly(val, offset)` / `warp_reduction_max/sum` | warp butterfly / reductions (no cross-warp) | `utils.py:318` |

**tcgen05 / TMEM**
| Symbol | Meaning | Ref |
|---|---|---|
| `tcgen05.MmaTF32Op / MmaF16BF16Op / MmaF8F6F4Op / MmaI8Op` | MMA ops (inst-K: TF32=8, F16=16, FP8/I8=32) | `mma.py` |
| `tcgen05.OperandSource.SMEM/.TMEM` | A-operand source (TMEM ⇒ A is K-major) | `mma.py` |
| `tcgen05.CtaGroup.ONE/.TWO` | 1-SM / 2-SM MMA | `mma.py` |
| `tcgen05.Field.ACCUMULATE` | `tiled_mma.set(...)` toggle (False=overwrite k0) | `blackwell_helpers.py:96` |
| `sm100_utils.make_trivial_tiled_mma(...)` | build tiled MMA (preferred over raw) | `blackwell_helpers.py` |
| `tcgen05.commit(mbar, mask, cta_group)` | UMMA-completion arrive (**wrap in elect_one**) | §9.4; `helpers.py::commit` |
| `tcgen05.Ld32x32bOp / St32x32bOp(Repetition(n))` | TMEM↔reg copy atoms | `tcgen05/copy.py` |
| `tcgen05.make_tmem_copy(atom, tmem_t)` | build the tiled TMEM copy | `dense_gemm.py` |
| `sm100_utils.get_tmem_load_op(...)` | auto-pick the t2r atom | `blackwell_helpers.py` |
| `utils.TmemAllocator(...).allocate/wait_for_alloc/retrieve_ptr/free` | TMEM lifecycle | `flash_fwd_sm100.py:885` |
| `cute.arch.get_max_tmem_alloc_cols("sm_100")` | = **512** (full acc ⇒ 1 CTA/SM; min 32, pow-2, ×32) | `cute/arch/tmem.py` |
| (wall) | acc tile pads to **NBP≥64 rows** → ~2.0× serial-tcgen05 floor at n≤512 | FA4 blueprint; project memory |

**TMA / cpasync**
| Symbol | Meaning | Ref |
|---|---|---|
| `make_tiled_tma_atom_A/B(op, t, smem_layout, mma_tiler, tiled_mma, cluster)` | MMA-coupled TMA atom | `sm100_hd256_2cta_fmha_forward.py:457` |
| `cpasync.make_tiled_tma_atom(op, t, smem_layout, tiler)` | generic TMA atom (e.g. S2G store) | `flash_fwd_sm100.py:618` |
| `cpasync.CopyBulkTensorTile{G2S,G2SMulticast,S2G}Op` | TMA load / multicast / store ops | `cpasync/__init__.py` |
| `cpasync.prefetch_descriptor(atom)` | warm descriptor (warp 0, once) | `flash_fwd_sm100.py:855` |
| `cpasync.tma_partition(atom, mcast_coord, cta_layout, group_modes(s), group_modes(g))` | partition for the copy | `flash_fwd_sm100.py:1415` |
| `cpasync.create_tma_multicast_mask(cluster_layout_vmnk, coord, mcast_mode)` | multicast CTA mask | `dense_gemm.py` |
| `sm100_utils.cluster_shape_to_tma_atom_A/B(...)` | pick op by SM-count/multicast | `blackwell_helpers.py` |
| `tcgen05.SmemLayoutAtomKind.{K_SW128, MN_SW128_32B, …}` | swizzled smem layout (128B; `MN_SW128_32B` for transposed) | `mma.py` |

**pipeline**
| Symbol | Meaning | Ref |
|---|---|---|
| `pipeline.PipelineTmaUmma.create(..., tx_count=)` | TMA-load → UMMA (commit = no-op, TMA arms barrier) | `sm100_hd256_2cta_fmha_forward.py:607` |
| `pipeline.PipelineUmmaAsync.create(...)` | UMMA → threads (TMEM/epilogue; commit via `tcgen05.commit`) | `sm100_hd256_2cta_fmha_forward.py:625` |
| `pipeline.PipelineAsyncUmma.create(...)` | **threads → UMMA** (manual-fill operand; commit = real arrive) | `sm100_hd256_2cta_fmha_forward.py:635` |
| `pipeline.PipelineTmaAsync / PipelineAsync / PipelineCpAsync` | TMA→threads / threads→threads / cp.async→threads | [pipeline doc](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html) |
| `.create(num_stages, producer_group, consumer_group, barrier_storage=, [tx_count=,] cta_layout_vmnk=, defer_sync=)` | constructor (tx_count only on TMA classes) | pipeline doc |
| `.make_participants()` | split one pipeline → `(producer, consumer)` handles | `sm100_hd256_2cta_fmha_forward.py:607` |
| `producer_acquire/commit`, `consumer_wait/release` | the 4 ops (acquire/wait block; commit/release don't) | [cpp pipeline](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) |
| `acquire_and_advance()/wait_and_advance()/.tail()` | FA4 handle-style drive (`.index`, `.barrier`) | `sm100_hd256_2cta_fmha_forward.py:947/1088/1030` |
| `pipeline.make_pipeline_state(PipelineUserType.Producer/.Consumer, stages)` | `(index, phase, count)`; `.advance()` | `flash_fwd_sm100.py:1357` |
| `pipeline.CooperativeGroup(Agent.Thread, size)` | arrival-count group (size = arrive count) | pipeline doc |
| `pipeline.producer_get_barrier(state)` | mbarrier ptr to hand to `cute.copy(tma_bar_ptr=)` | `copy_utils.py:363` |
| `_override_create(parent, child)` | re-class a frozen CUTLASS pipeline to graft methods | `pipeline.py:20` |

**utils / scheduler / hardware**
| Symbol | Meaning | Ref |
|---|---|---|
| `utils.HardwareInfo().get_device_multiprocessor_count()` | SM count (B200 ≈ 148) | `hardware_info.py` |
| `utils.HardwareInfo().get_max_active_clusters(sz)` | occupancy oracle (persistent grid sizing) | `hardware_info.py` |
| `utils.StaticPersistentTileScheduler` | grid-stride persistent scheduler | `tile_scheduler.py:287` |
| `WorkTileInfo.is_valid_tile / .tile_idx` | loop-exit flag / coord | `tile_scheduler.py:350` |
| `barrier.py::wait_eq/arrive_inc` + `dsl_user_op`+`llvm.inline_asm` | grep-clean cross-CTA barrier template | `barrier.py:55` |

> **Module map (4.5.2):** `cutlass.cute.arch` (intrinsics/mbarriers/fences), `cutlass.cute.nvgpu.{tcgen05, cpasync, warp}` (MMA/TMA/ldmatrix atoms), `cutlass.cute.math` (sqrt/rsqrt/exp2), `cutlass.pipeline` (Pipeline* + CooperativeGroup + make_pipeline_state), `cutlass.utils` (TmemAllocator, HardwareInfo, schedulers) + `cutlass.utils.blackwell_helpers as sm100_utils`. **Not in 4.5.2:** `warp_specialize` (Gluon-only → manual warp roles), `cutlass.cute.experimental` (needs CUDA ≥13.1; eval is 12.9). **(verify on B200: enumerate exact `cute.math` list, `if_generate`, `tcgen05.commit` DSL spelling via `dir()`/`inspect.signature` on the installed wheel.)**

---

## 10. Review notes — corrections & verify-on-B200

### 1. CORRECTIONS (wrong / contradicts research pool)

- **§2.7 `make_warp_uniform` listed as "REQUIRED before branching on warp_idx"** — overstated as established fact. The pool only has it as **plausible/empirical** ("the failure mode is the empirical claim"; §7.1 itself tags the omission-faults claim "verify on B200"). §2.7's flat "REQUIRED" contradicts §7.1's hedge. Make consistent: required-by-convention, omission-faults unconfirmed.

- **§2.7 `warp_reduction_max/_sum` / §9.6 `warp_reduction_max/sum`** — the verified DSL spelling from the pool is **`cute.arch.warp_reduction(...)` / `warp_reduction_max(...)` / `warp_reduction_sum(...)`** AND a distinct **`warp_redux_sync(value, kind, mask_and_clamp=...)`** (pipeline-sync pool). FA4 itself uses `cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)`. The draft's names are right but it omits the `threads_in_group=`/`mask_and_clamp` kwargs — add them; also `cute.arch.warp_reduction` (no suffix) exists.

- **§2.7 `cute.arch.barrier(barrier_id=None, number_of_threads=None)`** — pool confirms this signature, but the draft elsewhere (§6.6, §7.3) says NamedBarrier ID 0 is reserved for `sync_threads()`. The verified arch signature has `barrier_id=None` default (= unnamed CTA barrier). Clarify: `barrier()` with no id ≠ named-barrier-0; the "id 0 reserved" rule is about the *named* enum space.

- **§3.7 / §5.5 TMEM "256 KB/SM"** — pool says 128 lanes × 512 cols × 32-bit = **256 KB** in one place but the draft also writes "128 lanes × 512 columns of 32-bit = 256 KB" consistently; OK. However the draft repeatedly conflates the **NBP≥64 ROW wall** with the column-budget wall — the pool is explicit these are *two distinct* walls (rows: [NBP,BW] pads to 64 min; cols: 512 monopoly). §3.7 gets this right; §5.5 says "This is the 'NBP≥64 / 1-CTA-per-SM' wall behind…the column budget" — that sentence **fuses the two walls** the pool separates. Fix the §5.5 wording.

- **§5.1 `MmaTF32Op` constructor arg name `a_src`** — pool (`mma.py` VERIFIED) gives the positional `MmaTF32Op(instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)`. Draft matches. But §5.1 calls the enum `OperandSource.SMEM (= smem_desc, default)` — pool confirms `= "smem_desc"`. OK. No correction; flagging that `a_src` is **A-operand only** is correctly stated.

- **§9.6 / §5.1 `tcgen05.OperandMajorMode` "deprecated → use `cute.nvgpu.OperandMajorMode`"** — correct per pool, but §3.1 example and §9.6 index still list bare `OperandMajorMode.K` without the `cute.nvgpu.` qualifier in the s2t/make_trivial calls. Cosmetic but worth unifying to the non-deprecated spelling everywhere.

- **§4.0 / §9.6 `tcgen05.St16x{64,128,256}bOp`** — pool's verified store-atom list is `St16x64bOp, St16x128bOp, St16x256bOp, St16x32bx2Op, St32x32bOp`. Draft §4.6 table lists these correctly, but §4.0's decision table writes "`tcgen05.St32x32bOp / St16x{64,128,256}bOp`" — fine. No error; consistent.

### 2. UNCERTAIN — VERIFY ON B200

- **`tcgen05.commit(mbar, mask=None, cta_group=ONE)` DSL spelling** — used pervasively (§5.3, §5.6, §6.5, §8.4, §9.4) as if confirmed. Pool is explicit: the **exact DSL function name was NOT found verbatim** in docs/source; it's `cutlass::arch::umma_arrive` in C++, PTX `tcgen05.commit...mbarrier::arrive`. The project calls it `tcgen05.commit(mbar)` empirically. Every load-bearing use should carry the "verify spelling via `dir(cutlass.cute.nvgpu.tcgen05)`" caveat — draft does this in §6.6/§9.6 but NOT at the first/heaviest uses (§5.6, §8.4).

- **`PipelineAsyncUmma.producer_commit` emits a real arrive** — draft treats this as the fix in §6.4/§8.3/§9.3. Pool flags it **INFERRED** (the class name fits; the Python docstring mislabels it; C++ symmetry rule is the only authority). Draft already hedges — keep the hedge prominent; the empirical project path was the **hand-rolled ring**, not the class. Consider leading with the ring as the *validated* path and `PipelineAsyncUmma` as the *plausible* one (draft §6.5 ordering slightly undersells this).

- **`make_smem_layout_a/b` arg order** — draft uses `make_smem_layout_a(tiled_mma, mma_tiler, a_dtype, num_stages)` (§3.5) but FA4 pool shows `make_smem_layout_a(tiled_mma, mma_tiler, a_dtype, num_stages)` in some refs and the pool's blackwell_helpers list doesn't pin the order. Verify positional order on wheel.

- **`TmemAllocator` two signatures** — draft §3.7 correctly flags FA4-positional vs dense_gemm-kwarg both exist; pool confirms the kwarg form `TmemAllocator(arch=, is_two_cta=, num_allocated_columns=, two_cta_tmem_dealloc_mbar_ptr=)`. The FA4 positional form `(holding_buf, barrier_for_retrieve=, allocator_warp_id=, is_two_cta=)` is a **different constructor shape** — these may be version-skewed; verify which the 4.5.2 wheel exposes.

- **`fence_proxy("async.shared", space="cta")` as an alias for `fence_view_async_shared()`** — draft §4.8/§6.4 treats them as interchangeable. Pool lists both but does not confirm equivalence; gau-nernst uses `tcgen05.fence::after_thread_sync` as the pre-MMA fence, which may be a *third* distinct fence. Verify the three are not conflated.

- **`mbarrier_arrive(..., arrive_count=1)` kwarg** — pipeline-sync pool shows `mbarrier_arrive(mbar_ptr, peer_cta_rank_in_cluster=None, arrive_count=1)` but core-runtime pool shows only `mbarrier_arrive(mbar_ptr, peer_cta_rank_in_cluster=None)`. The `arrive_count` param is version-uncertain — verify.

- **`SmemLayoutAtomKind` enum on `tcgen05` vs `cute.nvgpu`** — draft §3.5 imports it from `tcgen05`, §5.1/§9.6 also. Pool confirms it's in the `tcgen05` enum list. OK, but `make_smem_layout_atom(kind, element_type)` arg order unverified — confirm.

- **`Repetition` as `tcgen05.Repetition(32)` vs `tcgen05.Repetition.x64`** — draft uses BOTH forms (§5.5 `Repetition.x64`, §4.6/§9.4 `Repetition(32)`). Pool shows `Repetition` with members `x1…x128`. The callable-int form `Repetition(32)` vs enum-member `Repetition.x64` may not both be valid — verify which the wheel accepts (this is a real footgun, not cosmetic).

### 3. GAPS (missing / too thin)

- **cp.async (non-bulk) producer ring** — §4 lists `CopyG2SOp` and commit/wait_group, but there is **no worked cp.async→compute pipeline example** despite the pool flagging `PipelineCpAsync` + `cp_async_commit_group()`/`cp_async_wait_group()` as the alternative fill path for the ring (§6.4 mentions it in one line). Add a minimal cp.async fill recipe — it's the documented option-(ii) the project flagged.

- **CUTLASS C++ mapping** — the title scopes "cute-DSL + CUTLASS" and the eval HAS nvcc + C++ CUTLASS, but the draft is ~100% Python with only scattered C++ atom-name cross-refs. Either add a short DSL↔C++ concept table (`CollectiveMma`, `cute::Swizzle<B,M,S>`, `TMEM::Allocator{1,2}Sm`, `UMMA::Layout_K_SW128_Atom`) or explicitly state C++ is out of scope. Currently neither (pool gap G9).

- **`HardwareInfo` SMEM-capacity query** — §7.5 hardcodes "228 KB/SM (verify)". Pool notes `HardwareInfo` has `get_l2_cache_size_in_bytes()` and occupancy probes but the draft never shows how to *query* per-SM smem vs hardcode. Thin.

- **`testing.benchmark` / `JitArguments`** — the core-runtime pool documents `cutlass.cute.testing.benchmark(compiled, workspace_generator=, warmup_iterations=, iterations=)` as the in-DSL timing harness. Given profiling is timing-ablation-only (ncu dead), this is directly relevant and **entirely absent** from the draft.

- **`cute.assume(x, divby=N)` for vectorization** — mentioned once in §2.4. The FA4 pool shows `assume_strides_aligned`/`assume_tensor_aligned` (`cute.assume(s, divby=128//width)`) as the idiom that *unlocks vectorized TMA loads* — a correctness/perf gate worth its own line, currently buried.

- **`recast_tensor` vs value `.bitcast`** — covered for the tf32x3 scalar split, but the *tensor-level* reinterpret (`cute.recast_tensor`) use cases (fp8-as-uint8 DLPack workaround from `cute_dsl_utils.py:62-84`) are not mentioned; that workaround is a real deployment gotcha for low-precision dtypes.

- **`cooperative=`/`use_pdl=` launch flags** — §2.1 lists them as decorator args but never warns that `cooperative=True` is a cooperative-launch path (adjacent to the banned stream/graph territory) and Programmatic Dependent Launch (`use_pdl`) interacts with persistence. One caution line needed.

### Strongest parts (keep as-is)
The tf32x3 bit-AND split + 3 dead alternatives (§5.6/§8.2/§9.5), the `PipelineTmaUmma.producer_commit`-is-`pass` deadlock diagnosis (§6.1/§6.4/§8.3), the hand-rolled mbarrier ring with elect-one `tcgen05.commit` + prime + init-once/no-reinit heisenbug (§6.5/§8.5/§9.4), the four-fence table (§4.8/§9.2), and the elect-one asymmetry (`cute.gemm` no-elect vs `tcgen05.commit` must-elect) are all faithful to the pool, correctly hedged where the pool hedges, and are the highest-value content. The §3.9/§8/§9 per-section "verify-on-B200" confidence lines are exactly the right discipline for a scarce-info audience.