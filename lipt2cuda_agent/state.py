"""State schema for LIPT2CudaAgent workflow."""
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List
from pathlib import Path
from scripts.individual import KernelIndividual


class LIPT2CudaAgentState(TypedDict, total=False):
    """State for the LIPT2CudaAgent LangGraph workflow.

    This state tracks the complete workflow of generating, benchmarking,
    and optimizing CUDA kernels through multiple rounds.
    """

    # Task metadata
    task_path: Path
    task_name: str
    round_idx: int
    max_rounds: int

    # Current iteration state
    current_kernel: Optional[KernelIndividual]
    current_code: str
    current_metrics: Dict[str, Any]
    is_runnable: bool

    # Best kernel tracking
    best_kernel: Optional[KernelIndividual]
    best_score: float

    # History tracking
    score_history: List[float]
    error_flags: List[bool]
    last_score: float

    # Directories
    code_dir: Path
    eval_dir: Path
    io_dir: Path
    log_path: Path
    fig_dir: Path

    # Configuration
    args: Any  # argparse.Namespace

    # Phase control
    phase: str  # "seed", "repair", "optimize"
    next_action: str  # Used by routers to determine next node

    # Intermediate outputs
    judger_strategy: Optional[Dict[str, Any]]
    error_log: str
    ncu_metrics_block: str

    # LLM callback
    call_llm: Any  # Callable for LLM queries

    # Test kernel path (for NCU profiling)
    test_kernel_path: Path
