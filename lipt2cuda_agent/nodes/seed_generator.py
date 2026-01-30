"""Seed generator node for LIPT2CudaAgent."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from lipt2cuda_agent.state import LIPT2CudaAgentState
from prompts.generate_custom_cuda import build_seed_prompt
from utils.kernel_io import extract_code_block, save_kernel_code
from scripts.individual import KernelIndividual


def seed_generator_node(state: LIPT2CudaAgentState) -> LIPT2CudaAgentState:
    """Generate the initial kernel (Round 0).

    This node:
    1. Builds the seed prompt from the task architecture
    2. Calls the LLM to generate initial CUDA code
    3. Extracts and saves the kernel code
    4. Creates a KernelIndividual
    5. Updates state for benchmarking

    Args:
        state: Current workflow state

    Returns:
        Updated state with current_kernel set
    """
    print("[Seed] Generating the initial kernel ...")

    # Build seed prompt
    seed_prompt = build_seed_prompt(
        arch_path=state["task_path"],
        gpu_name=state["args"].gpu
    )

    # Save prompt to io_dir
    prompt_file = state["io_dir"] / f"round{state['round_idx']:03d}_seed_prompt.txt"
    prompt_file.write_text(seed_prompt, encoding="utf-8")

    # Call LLM
    call_llm = state["call_llm"]
    raw = call_llm(
        seed_prompt,
        sys_prompt=None,  # Uses default_system_prompt
        log_path=state["log_path"],
        call_type="seed",
        round_idx=state["round_idx"],
    )

    # Save raw reply
    reply_file = state["io_dir"] / f"{state['round_idx']}_raw_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")

    # Extract code
    code = extract_code_block(raw) or raw

    # Save kernel code
    code_path = save_kernel_code(code, state["code_dir"])

    # Create KernelIndividual
    ind = KernelIndividual(code)
    ind.code_path = code_path

    # Update state
    return {
        **state,
        "current_kernel": ind,
        "current_code": code,
        "phase": "seed",
        "next_action": "benchmark",
    }
