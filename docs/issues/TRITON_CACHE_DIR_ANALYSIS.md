# Triton cache directory not updating (~/.cache/triton)

## Phenomenon

After setting `SGLANG_TRITON_CACHE_DIR` default to `~/.cache/triton`, the directory was not updated when EP/scheduler ran Triton JIT (e.g. `alloc_extend` → ptxas → cubin). The compiled cubin was not written under `~/.cache/triton`.

## Cause

Triton uses the **environment variable `TRITON_CACHE_DIR`** to decide where to write the cache. If it is not set, Triton falls back to **`~/.triton/cache`**.

SGLang only set `TRITON_CACHE_DIR` in one path:

- **Engine / HTTP path**: `_launch_subprocesses()` → `_set_envs_and_config()` sets `TRITON_CACHE_DIR` before launching scheduler processes. So when using `launch_server()` (HTTP) or `Engine`, the scheduler child processes **do** inherit `TRITON_CACHE_DIR` and write to `~/.cache/triton`.

- **gRPC path**: `launch_scheduler_process_only()` in `grpc/scheduler_launcher.py` is used by the gRPC server. It **does not** call `_set_envs_and_config()`. So when scheduler processes (or the DataParallelController process) are started, the parent process never set `TRITON_CACHE_DIR`, and children inherit the default (unset). Triton then uses **`~/.triton/cache`**, so `~/.cache/triton` never gets updated.

- **PD / custom launchers**: If decode (or prefill) scheduler processes are started by a script or another entry point that does not set `TRITON_CACHE_DIR` before `proc.start()`, the same situation occurs: Triton uses `~/.triton/cache`.

## When Triton reads the cache dir

Triton resolves the cache directory from `os.environ` when it needs to read/write the cache (typically at first JIT compile). The **scheduler process** must therefore see `TRITON_CACHE_DIR` in its environment at process start. That only happens if the **parent** sets `os.environ["TRITON_CACHE_DIR"]` **before** calling `proc.start()`.

## Fix (applied)

1. **environ.py**  
   - Added `SGLANG_TRITON_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/triton"))`.

2. **engine.py**  
   - In `_set_envs_and_config()`, at the start: if `TRITON_CACHE_DIR` is not set, set it to `envs.SGLANG_TRITON_CACHE_DIR.get()`.  
   - So Engine/HTTP path continues to use `~/.cache/triton`.

3. **grpc/scheduler_launcher.py**  
   - In `launch_scheduler_process_only()`, right after `server_args.check_server_args()` and before any `proc.start()`: if `TRITON_CACHE_DIR` is not set, set it to `~/.cache/triton`.  
   - So both direct scheduler processes and the DataParallelController process (and its scheduler children) inherit `TRITON_CACHE_DIR`, and Triton writes to `~/.cache/triton`.

## Verification

- **Engine/HTTP**: Start with the usual HTTP entry; trigger alloc_extend (e.g. first decode prealloc). Check that `~/.cache/triton` gets new hash subdirs.
- **gRPC**: Start with the gRPC entry; same trigger. Again check `~/.cache/triton`.
- If you still see no updates under `~/.cache/triton`, check **`~/.triton/cache`**; if that is updating, the process you are watching is still started without `TRITON_CACHE_DIR` (e.g. a custom or PD launcher). Set `TRITON_CACHE_DIR=~/.cache/triton` (or `SGLANG_TRITON_CACHE_DIR`) in that launcher’s environment before starting the scheduler processes.
