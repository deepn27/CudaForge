"""Routing logic for LIPT2CudaAgent workflow."""
from __future__ import annotations
from typing import Literal

from lipt2cuda_agent.state import LIPT2CudaAgentState


def route_after_benchmark(
    state: LIPT2CudaAgentState
) -> Literal["update_state", "END"]:
    """Route after benchmark: check if max rounds reached.

    Args:
        state: Current workflow state

    Returns:
        "update_state" to continue, "END" to finish
    """
    # Check if we've reached max rounds
    if state["round_idx"] >= state["max_rounds"] - 1:
        return "END"
    return "update_state"


def route_repair_or_optimize(
    state: LIPT2CudaAgentState
) -> Literal["correctness_judger", "performance_judger"]:
    """Route based on kernel runnable status.

    Args:
        state: Current workflow state

    Returns:
        "correctness_judger" if not runnable (repair path),
        "performance_judger" if runnable (optimize path)
    """
    if not state.get("is_runnable", False):
        return "correctness_judger"
    return "performance_judger"


def update_state_node(state: LIPT2CudaAgentState) -> LIPT2CudaAgentState:
    """Update state after each round: track scores, errors, best kernel.

    This node:
    1. Increments round_idx
    2. Updates score_history and error_flags
    3. Tracks best_kernel and best_score
    4. Manages last_score for curve plotting

    Args:
        state: Current workflow state

    Returns:
        Updated state for next round
    """
    ind = state.get("current_kernel")
    runnable = state.get("is_runnable", False)

    # Get current score
    this_score = ind.score if (ind and ind.score is not None and runnable) else None

    # Update history
    score_history = state.get("score_history", [])
    error_flags = state.get("error_flags", [])
    last_score = state.get("last_score", 0.0)
    best_score = state.get("best_score", 0.0)
    best_kernel = state.get("best_kernel")

    if this_score is not None:
        last_score = this_score
        score_history.append(this_score)
        error_flags.append(False)

        # Update best
        if this_score > best_score:
            best_score = this_score
            best_kernel = ind

            # Update test_kernel.py with best kernel
            try:
                with open(state["test_kernel_path"], "w") as f:
                    f.write(best_kernel.code)
            except Exception as e:
                print(f"WARNING: failed to update test_kernel.py: {e}")
    else:
        # On failure: keep last score and mark error
        score_history.append(last_score)
        error_flags.append(True)

    # Increment round
    next_round = state["round_idx"] + 1

    # Print round progress
    task_name = state.get("task_name", "unknown")
    print(f"\n[{task_name}] Round {next_round}")

    return {
        **state,
        "round_idx": next_round,
        "score_history": score_history,
        "error_flags": error_flags,
        "last_score": last_score,
        "best_score": best_score,
        "best_kernel": best_kernel,
        "next_action": "route_phase",
    }
