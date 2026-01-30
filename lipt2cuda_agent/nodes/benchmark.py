"""Benchmark node for LIPT2CudaAgent."""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional

from lipt2cuda_agent.state import LIPT2CudaAgentState
from scripts.individual import KernelIndividual


def _bench_worker_entry(test_py: str,
                        ref_py: str,
                        device_idx: int,
                        warmup: int,
                        repeat: int,
                        tol: float,
                        conn) -> None:
    """
    Subprocess entry: set GPU, call compare_and_bench, and send result or error
    back to the parent via a Pipe.
    """
    import torch
    from pathlib import Path
    from utils.compile_and_run import CompilationError, AccuracyError, compare_and_bench

    def _sanitize_error_message(exc: Exception) -> str:
        """Strip pybind's large tensor printouts."""
        msg = str(exc)
        if "Invoked with:" in msg:
            msg = msg.split("Invoked with:", 1)[0].rstrip()
        return msg

    def _last_n_lines(text: str, n: int = 150) -> str:
        lines = text.splitlines()
        return "\n".join(lines[-n:]) if len(lines) > n else text

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)

        res = compare_and_bench(
            ref_py=Path(ref_py),
            test_py=Path(test_py),
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        conn.send(("ok", res))
    except Exception as e:
        try:
            cleaned = _sanitize_error_message(e)
            msg = _last_n_lines(cleaned)
        except Exception:
            msg = str(e)

        if isinstance(e, CompilationError):
            err_type = "CompilationError"
        elif isinstance(e, AccuracyError):
            err_type = "AccuracyError"
        else:
            err_type = e.__class__.__name__

        conn.send(("err", {"type": err_type, "message": msg}))
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


def benchmark_node(state: LIPT2CudaAgentState) -> LIPT2CudaAgentState:
    """Benchmark the current kernel with subprocess isolation.

    This node:
    1. Runs the kernel in a spawned subprocess to isolate CUDA context
    2. Updates metrics and score in the KernelIndividual
    3. Saves metrics to eval_dir
    4. Copies successful kernels to test_kernel.py for NCU profiling
    5. Updates state with runnable status and metrics

    Args:
        state: Current workflow state

    Returns:
        Updated state with metrics and runnable status
    """
    import torch
    from multiprocessing import get_context

    ind = state["current_kernel"]
    if ind is None:
        raise ValueError("current_kernel is None in benchmark_node")

    phase = state.get("phase", "unknown")
    args = state["args"]

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Spawn subprocess
    p = ctx.Process(
        target=_bench_worker_entry,
        args=(
            str(ind.code_path),
            str(state["task_path"]),
            args.device,
            args.warmup,
            args.repeat,
            args.tol,
            child_conn,
        ),
    )
    p.start()

    try:
        child_conn.close()
    except Exception:
        pass

    # Wait for result
    p.join()
    payload = parent_conn.recv() if parent_conn.poll() else None

    try:
        parent_conn.close()
    except Exception:
        pass

    # Process result
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase
            speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            print(f"[{phase}] score={speedup:.4f}")

            # Copy successful kernel to test_kernel.py for NCU profiling
            try:
                test_kernel_path = state["test_kernel_path"]
                src = Path(ind.code_path)
                if src.exists():
                    shutil.copy2(src, test_kernel_path)
                    print(f"[{phase}] saved successful kernel to: {test_kernel_path}")
            except Exception as copy_exc:
                print(f"[{phase}] WARNING: failed to save test_kernel.py: {copy_exc}")

        else:
            err_type = "RuntimeError"
            message = data
            if isinstance(data, dict):
                err_type = data.get("type", err_type) or err_type
                message = data.get("message", message)

            if not isinstance(message, str):
                message = str(message)

            print(f"\033[91mTest Error ({err_type}):\033[0m {message}")
            ind.metrics = {
                "runnable": False,
                "phase": phase,
                "error_type": err_type,
                "message": message,
            }
            ind.score = float("-inf")
            print(f"[{phase}] failed. See metrics.message for details.")
    else:
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": "SubprocessCrashed",
            "message": "subprocess exited unexpectedly (no payload received)",
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. Subprocess crashed.")

    # Save metrics
    try:
        saved = ind.save_metrics(state["eval_dir"])
        print(f"[{phase}] metrics saved to: {saved}")
    except Exception as save_exc:
        print(f"[{phase}] WARNING: failed to save metrics: {save_exc}")

    # Cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(args.device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # Update state
    is_runnable = bool(ind.metrics.get("runnable", False))
    current_metrics = ind.metrics or {}

    return {
        **state,
        "current_kernel": ind,
        "current_metrics": current_metrics,
        "is_runnable": is_runnable,
        "next_action": "route",
    }
