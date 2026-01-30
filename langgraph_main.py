"""LIPT2CudaAgent main entry point using LangGraph."""
from __future__ import annotations
import asyncio
import argparse
import re
import random
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda import default_system_prompt
from lipt2cuda_agent.graph import build_lipt2cuda_graph
from lipt2cuda_agent.checkpointing import (
    create_checkpoint_saver,
    get_checkpoint_path,
    get_thread_id,
)
from lipt2cuda_agent.state import LIPT2CudaAgentState


# ------------------------- CLI -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("LIPT2CudaAgent: LangGraph-based kernel generation/optimization")
    p.add_argument(
        "arch_py",
        type=Path,
        help="Path to a single task .py file OR a directory containing many tasks (.py)",
    )
    p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="local", help="LLM provider (local, openai, deepseek, vllm, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=8000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="deepseek-ai/deepseek-coder-6.7b-instruct", help="LLM model")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations per task")
    p.add_argument("--work_dir", type=Path, default=Path("run"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-3, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=16384, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")

    # Anthropic-specific (for reasoning models like Claude)
    p.add_argument("--budget_tokens", type=int, default=10000, help="Budget tokens for Anthropic reasoning models (default: 10000)")
    p.add_argument("--is_reasoning_model", action="store_true", help="Enable reasoning mode for Anthropic models")

    # Multi-task controls
    p.add_argument("--first_n", type=int, default=0, help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument("--num_tasks", type=int, default=1, help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")

    p.add_argument("--subproc_id", type=int, default=0, help="Identifier for sub-process (e.g., when running multiple in parallel)")

    # LangGraph-specific
    p.add_argument("--enable_checkpointing", action="store_true", help="Enable SQLite checkpointing")
    p.add_argument("--checkpoint_dir", type=Path, default=Path(".lipt2cuda"), help="Directory for checkpoints")

    return p


# ---------------------- naming helpers -----------------
def _slugify_tag(text: str, max_len: int = 80) -> str:
    """Collapse a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if max_len > 0:
        slug = slug[:max_len]
    return slug or "unknown"


def _build_run_tag(server_type: str, model_name: str) -> str:
    server_tag = _slugify_tag(server_type)
    model_tag = _slugify_tag(model_name)
    return f"{server_tag}_{model_tag}"


# ---------------------- small utils --------------------
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------- task helpers -------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """If a directory, return all .py files (sorted); if a file, return [file]."""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found")


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _sample_tasks(all_tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not all_tasks:
        raise RuntimeError("No .py tasks found.")
    k = max(1, min(k, len(all_tasks)))
    if seed is None or seed == 0:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(all_tasks, k)


def _plot_scores(save_path: Path, scores: List[float], err_flags: List[bool], title: str):
    """Plot per-round score curve; mark error rounds with an 'x'."""
    xs = list(range(len(scores)))
    plt.figure()
    plt.plot(xs, scores, marker="o")
    for x, y, bad in zip(xs, scores, err_flags):
        if bad:
            plt.scatter([x], [y], marker="x")
    plt.xlabel("Round")
    plt.ylabel("Speedup (ref/test)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _append_usage_totals(log_path: Path) -> Dict[str, int]:
    """Append a totals row to usage.csv and return the summed token counts."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not log_path.exists():
        return totals

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames or not rows:
        return totals

    for row in rows:
        if row.get("call_type") == "sum" or row.get("timestamp") == "Total":
            continue
        for key in totals:
            try:
                totals[key] += int(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue

    total_row = {fn: "" for fn in fieldnames}
    for key, value in totals.items():
        if key in total_row:
            total_row[key] = str(value)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(total_row)

    return totals


# ------------------- LLM caller factory ----------------
def _make_llm_caller(args):
    """Create LLM caller function with args bound."""
    def call_llm(
        prompt: str,
        sys_prompt: Optional[str] = None,
        log_path: Optional[Path] = None,
        call_type: str = "unknown",
        round_idx: int = -1,
    ) -> str:
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        res = query_server(
            prompt=prompt,
            system_prompt=sp,
            server_type=args.server_type,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            server_address=args.server_address,
            server_port=args.server_port,
            budget_tokens=args.budget_tokens,
            is_reasoning_model=args.is_reasoning_model,
            log_path=str(log_path) if log_path else None,
            call_type=call_type,
            round_idx=round_idx,
        )
        if isinstance(res, list):
            return res[0] if res else ""
        return str(res)
    return call_llm


# --------------------- single-task run -----------------
async def _run_single_task_async(
    task_path: Path,
    args,
    batch_dir: Path,
    batch_timestamp: str,
) -> Dict[str, Any]:
    """Run a single task using LangGraph workflow."""
    # Per-task directories
    task_root = (batch_dir / task_path.stem).resolve()
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"

    # Write task to ref.py
    root_dir = Path(__file__).resolve().parent
    ref_py = root_dir / f"ref_{args.subproc_id}.py"
    test_kernel_path = root_dir / f"test_kernel_{args.subproc_id}.py"
    bench_ref_inputs_path = root_dir / f"bench_ref_inputs_{args.subproc_id}.py"

    content = task_path.read_text(encoding="utf-8")
    with open(ref_py, "w", encoding="utf-8") as f:
        f.write(content)

    # Also write to bench_ref_inputs for NCU profiling
    with open(bench_ref_inputs_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Create LLM caller
    call_llm = _make_llm_caller(args)

    # Initialize state
    initial_state: LIPT2CudaAgentState = {
        "task_path": task_path,
        "task_name": task_path.stem,
        "round_idx": 0,
        "max_rounds": args.round,
        "current_kernel": None,
        "current_code": "",
        "current_metrics": {},
        "is_runnable": False,
        "best_kernel": None,
        "best_score": float("-inf"),
        "score_history": [],
        "error_flags": [],
        "last_score": 0.0,
        "code_dir": code_dir,
        "eval_dir": eval_dir,
        "io_dir": io_dir,
        "log_path": log_path,
        "fig_dir": fig_dir,
        "args": args,
        "phase": "seed",
        "next_action": "seed",
        "judger_strategy": None,
        "error_log": "",
        "ncu_metrics_block": "",
        "call_llm": call_llm,
        "test_kernel_path": test_kernel_path,
    }

    # Create checkpointer if enabled
    checkpointer = None
    thread_id = None
    if args.enable_checkpointing:
        checkpoint_path = get_checkpoint_path(task_path.stem, args.checkpoint_dir)
        checkpointer = create_checkpoint_saver(checkpoint_path)
        thread_id = get_thread_id(task_path.stem, batch_timestamp, 0)
        print(f"[Checkpointing] Enabled (in-memory) at: {checkpoint_path}")
        print(f"[Checkpointing] Thread ID: {thread_id}")

    # Build graph
    app = build_lipt2cuda_graph(checkpointer=checkpointer)

    # Run graph
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    print(f"[{task_path.name}] Starting LangGraph workflow...")

    final_state = None
    async for state in app.astream(initial_state, config):
        # Track state updates
        final_state = state

    # Extract final state (last node output)
    if final_state:
        # Get the actual state from the last update
        node_name = list(final_state.keys())[-1]
        final_state = final_state[node_name]

    # Plot results
    scores = final_state.get("score_history", [])
    err_flags = final_state.get("error_flags", [])
    best_score = final_state.get("best_score", float("-inf"))
    best_kernel = final_state.get("best_kernel")

    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    usage_totals = _append_usage_totals(log_path)

    return {
        "task": str(task_path),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(best_kernel.metrics.get("runnable", False)) if best_kernel and best_kernel.metrics else False,
        "task_dir": str(task_root),
        "figure": str(fig_path),
        "input_tokens_sum": usage_totals["input_tokens"],
        "output_tokens_sum": usage_totals["output_tokens"],
        "total_tokens_sum": usage_totals["total_tokens"],
    }


# --------------------- summary saving ------------------
def _save_global_summary(batch_dir: Path, summary: List[Dict[str, Any]], avg_speedup: float, accuracy: float, total_tokens_sum: float) -> None:
    """Save summary.json and summary.csv under the batch_dir."""
    batch_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "total_tokens_sum": total_tokens_sum,
        "num_tasks": len(summary),
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "best_score", "best_runnable", "task_dir", "figure"])
        for s in summary:
            writer.writerow([s["task"], f'{s["best_score"]:.6f}', int(
                bool(s["best_runnable"])), s["task_dir"], s["figure"]])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])
        writer.writerow(["total_tokens_sum", f"{int(total_tokens_sum)}"])

    print(f"[GLOBAL] Saved: {batch_dir/'summary.json'}")
    print(f"[GLOBAL] Saved: {csv_path}")


# --------------------------- main ----------------------
async def main_async():
    args = _build_arg_parser().parse_args()

    all_tasks = _collect_tasks(args.arch_py)

    # Create batch folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _build_run_tag(args.server_type, args.model_name)

    if args.arch_py.is_file():
        batch_name = f"{stamp}_{args.arch_py.stem}_{run_tag}"
    else:
        pick_note = f"first{args.first_n}" if (args.first_n and args.first_n > 0) else f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{stamp}_batch_{pick_note}_{run_tag}"

    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    # Single file
    if args.arch_py.is_file():
        res = await _run_single_task_async(all_tasks[0], args, batch_dir, stamp)
        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        total_tokens_sum = res.get("total_tokens_sum", 0)
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        return

    # Directory: first_n or sample
    if args.first_n and args.first_n > 0:
        picked = _pick_first_n(all_tasks, args.first_n)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
    else:
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        res = await _run_single_task_async(task, args, batch_dir, stamp)
        summary.append(res)

    # Global summary
    runnable_count = sum(1 for s in summary if s["best_runnable"])
    avg_speedup = sum(s["best_score"] for s in summary) / len(summary) if summary else 0.0
    accuracy = runnable_count / len(summary) if summary else 0.0
    total_tokens_sum = sum(s.get("total_tokens_sum", 0) for s in summary)

    print("\n" + "=" * 50)
    print(f"[GLOBAL] Avg speedup: {avg_speedup:.4f}")
    print(f"[GLOBAL] Accuracy: {accuracy:.4f} ({runnable_count}/{len(summary)})")
    print(f"[GLOBAL] Total tokens: {int(total_tokens_sum)}")
    print("=" * 50)

    _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
