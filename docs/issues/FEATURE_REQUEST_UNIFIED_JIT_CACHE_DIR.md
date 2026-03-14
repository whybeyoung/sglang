# Feature Request: Unified JIT / Precompilation Cache Directory

## Summary

**Request:** Unify all JIT and precompilation cache paths under a single configurable root so that users and operators can manage cache location, size, and persistence in one place. The current situation is fragmented across multiple env vars and default paths (including `/tmp`), which makes it hard to reason about where compiled artifacts live and to reuse caches across runs or machines.

## Current State (Fragmented)

### 1. Triton JIT cache

| What | Env var / source | Default / behavior |
|------|------------------|--------------------|
| Direct `@triton.jit` (e.g. allocator, MoE kernels) | `TRITON_CACHE_DIR` (Triton runtime) | Triton default: `~/.triton/cache`; often overwritten by Inductor to a path under `/tmp/torchinductor_*` |
| SGLang override | `SGLANG_TRITON_CACHE_DIR` (if implemented) | e.g. `~/.triton/cache` or `~/.cache/triton` |

- Set only in some entry paths (engine, gRPC launcher, scheduler process); custom/PD launchers may not set it.
- When PyTorch Inductor runs first, it can set `TRITON_CACHE_DIR` to its own subdir, so later Triton JIT (including allocator) writes under `/tmp`.

### 2. PyTorch Inductor (torch.compile)

| What | Env var / source | Default / behavior |
|------|------------------|--------------------|
| Inductor cache | `TORCHINDUCTOR_CACHE_DIR` (PyTorch) | `/tmp/torchinductor_<user>` (or similar) |
| SGLang override | `SGLANG_TORCHINDUCTOR_CACHE_DIR` (if implemented) | e.g. `~/.cache/sglang/inductor` |

- Default lives in `/tmp`, so cache is often non-persistent and can conflict with other users on shared nodes.

### 3. SGLang torch.compile (SGLangBackend)

| What | Env var / source | Default / behavior |
|------|------------------|--------------------|
| Base cache dir | `SGLANG_CACHE_DIR` (used in `compilation/backend.py`) | `~/.cache/sglang/` |
| Actual path | Derived | `{SGLANG_CACHE_DIR}/torch_compile_cache/<hash>/rank_*_*/`; inside that, `inductor_cache` and `triton_cache` subdirs are set for that compile. |

- Only used when `--enable-torch-compile` and the model is compiled via SGLangBackend. Default Inductor (other `@torch.compile(backend="inductor")` usages) still use `TORCHINDUCTOR_CACHE_DIR`.

### 4. DeepGEMM JIT cache

| What | Env var / source | Default / behavior |
|------|------------------|--------------------|
| DeepGEMM cache | `SGLANG_DG_CACHE_DIR` / `DG_JIT_CACHE_DIR` | `~/.cache/deep_gemm` |

- Set in `layers/deep_gemm_wrapper/compile_utils.py`; separate from Triton/Inductor.

### 5. Other cache-related (for context only)

- **P2P / custom all-reduce:** `SGLANG_CACHE_ROOT` = `~/.cache/sglang` (hardcoded in `custom_all_reduce_utils.py`) for a JSON file.
- **Diffusion:** `SGLANG_DIFFUSION_CACHE_ROOT` = `~/.cache/sgl_diffusion`.
- **HiCache:** `SGLANG_HICACHE_*_STORAGE_DIR` (storage backends, not JIT).
- **Outlines:** `SGLANG_DISABLE_OUTLINES_DISK_CACHE`.

These are not JIT/precompilation caches but show that “cache” naming is already spread across many env vars.

## Problems

1. **No single root:** Triton, Inductor, SGLang torch_compile, and DeepGEMM each have their own env or default; some write to `/tmp`, others to `~/.cache/...`.
2. **Order / overwrite:** Whoever sets `TRITON_CACHE_DIR` or `TORCHINDUCTOR_CACHE_DIR` first (e.g. Inductor) can override SGLang’s choice, so behavior depends on import and execution order.
3. **Operational burden:** Users cannot “point SGLang to one cache directory” for all JIT; copying or cleaning cache requires touching multiple locations.
4. **Reproducibility / migration:** Hard to document “copy this folder to another machine” when JIT output is under `/tmp`, `~/.triton`, `~/.cache/sglang`, and `~/.cache/deep_gemm`.

## Proposal

### 1. Introduce a single JIT cache root

- **New env var:** `SGLANG_JIT_CACHE_ROOT` (or `SGLANG_CACHE_ROOT` if we want to align with existing `SGLANG_CACHE_ROOT` in custom_all_reduce_utils).
- **Default:** `~/.cache/sglang` (or `$XDG_CACHE_HOME/sglang` when set).
- **Semantics:** All JIT/precompilation caches that SGLang controls should live under this root in fixed subdirs.

### 2. Standard layout under the root

Suggested subdirs (all under `SGLANG_JIT_CACHE_ROOT`):

| Subdir | Purpose | Maps from |
|--------|---------|-----------|
| `triton/` | Triton JIT (direct `@triton.jit` and, when possible, Triton used by Inductor) | `TRITON_CACHE_DIR` |
| `inductor/` | PyTorch Inductor (torch.compile) | `TORCHINDUCTOR_CACHE_DIR` |
| `torch_compile/` | SGLang torch.compile cache (hash-based, when using SGLangBackend) | `SGLANG_CACHE_DIR` + `torch_compile_cache` |
| `deep_gemm/` | DeepGEMM JIT | `SGLANG_DG_CACHE_DIR` / `DG_JIT_CACHE_DIR` |

- If we keep backward compatibility, existing env vars (`SGLANG_TRITON_CACHE_DIR`, `SGLANG_TORCHINDUCTOR_CACHE_DIR`, `SGLANG_CACHE_DIR`, `SGLANG_DG_CACHE_DIR`) could override the default subdir path when set; otherwise they are derived as `{SGLANG_JIT_CACHE_ROOT}/{subdir}`.

### 3. Single point of initialization

- **Early in process** (e.g. engine `_set_envs_and_config`, gRPC launcher before any `proc.start()`, and at the very start of `run_scheduler_process`):
  - If `SGLANG_JIT_CACHE_ROOT` is set, compute and set:
    - `TRITON_CACHE_DIR` = `{root}/triton` (unless already set)
    - `TORCHINDUCTOR_CACHE_DIR` = `{root}/inductor` (unless already set)
    - Optionally set `SGLANG_CACHE_DIR` = `{root}` so that `torch_compile_cache` becomes `{root}/torch_compile_cache/...`
    - `SGLANG_DG_CACHE_DIR` (or `DG_JIT_CACHE_DIR`) = `{root}/deep_gemm` (unless already set)
  - Create these directories if needed.
- This ensures one env var controls the base path and nothing later (e.g. Inductor) overwrites it unless we explicitly allow overrides.

### 4. Documentation and backward compatibility

- Document `SGLANG_JIT_CACHE_ROOT` (and the subdir layout) in the environment variables reference.
- Keep existing per-cache env vars as overrides: if `TRITON_CACHE_DIR` is already set, do not overwrite it from the root; same for others. This preserves existing scripts and launchers that set specific dirs.

## Benefits

- **One knob:** Set `SGLANG_JIT_CACHE_ROOT=/data/sglang_jit_cache` (or similar) and all JIT caches go under that path.
- **No `/tmp` by default:** Avoids non-persistent and user-specific defaults like `/tmp/torchinductor_<user>`.
- **Easier ops:** Clean, copy, or quota one tree; same for “copy cache to another machine” workflows.
- **Predictable layout:** Clear mapping from subdir to component (Triton, Inductor, torch_compile, DeepGEMM).

## Scope

- **In scope:** Triton, TorchInductor, SGLang torch.compile cache, DeepGEMM JIT cache; env var design and early initialization in engine/gRPC/scheduler.
- **Out of scope (for this request):** HiCache storage dirs, diffusion cache root, P2P JSON path, and other non-JIT caches; those can be unified in a separate effort if desired.

## References

- `python/sglang/srt/environ.py` – existing env vars
- `python/sglang/srt/entrypoints/engine.py` – `_set_envs_and_config`
- `python/sglang/srt/grpc/scheduler_launcher.py` – launcher env
- `python/sglang/srt/managers/scheduler.py` – `run_scheduler_process`
- `python/sglang/srt/compilation/compiler_interface.py` – `initialize_cache`, TORCHINDUCTOR/TRITON
- `python/sglang/srt/compilation/backend.py` – `SGLANG_CACHE_DIR`, torch_compile_cache
- `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` – DG_JIT_CACHE_DIR
- `docs/issues/TRITON_WARMUP_AND_CUBIN_SIZE_ANALYSIS.md` – why cache dirs get overwritten
- `docs/references/environment_variables.md` – current cache-related vars
