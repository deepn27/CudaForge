"""Performance judger node for LIPT2CudaAgent."""
from __future__ import annotations

from lipt2cuda_agent.state import LIPT2CudaAgentState
from prompts.judger_optimization import build_judger_optimization_prompts
from utils.kernel_io import extract_json, extract_cuda_kernel_names
from run_ncu import profile_bench, load_ncu_metrics, metrics_to_prompt


def performance_judger_node(state: LIPT2CudaAgentState) -> LIPT2CudaAgentState:
    """Analyze performance with NCU and generate optimization strategy.

    This node:
    1. Extracts CUDA kernel names from current kernel
    2. Runs NCU profiler to collect metrics
    3. Loads and formats NCU metrics
    4. Calls LLM to analyze metrics and suggest optimizations
    5. Extracts JSON optimization strategy
    6. Updates state with strategy for optimization node

    Args:
        state: Current workflow state

    Returns:
        Updated state with judger_strategy and ncu_metrics_block
    """
    print("[Optimization] Analyzing performance with NCU ...")

    current_kernel = state["current_kernel"]
    if current_kernel is None:
        raise ValueError("current_kernel is None in performance_judger_node")

    args = state["args"]

    # Extract kernel names                                                                                                                                                                                                                                                           
    test_kernel_path = state["test_kernel_path"]                                                                                                                                                                                                                                      
    kernel_names = extract_cuda_kernel_names(test_kernel_path)
    print(f"Detected kernel names: {kernel_names}")

    # Profile with NCU
    csv_path = profile_bench(
        bench_py=f"bench_ref_inputs_{args.subproc_id}.py",
        out_csv=f"ncu_temp_{args.subproc_id}.csv"
    )

    # Load metrics
    metrics_df = load_ncu_metrics(
        csv_path,
        extra_keep=("Kernel Name",),
        name_list=kernel_names,
        select="last"
    )

    # Format metrics for prompt
    metrics_block = metrics_to_prompt(metrics_df)

    # Build judger prompts
    sys_judge_prompt, judge_prompt = build_judger_optimization_prompts(
        arch_path=state["task_path"],
        gpu_name=args.gpu,
        ncu_metrics_block=metrics_block,
        cuda_code=current_kernel.code,
    )

    # Save prompt
    prompt_file = state["io_dir"] / f"round{state['round_idx']:03d}_judge_optimization_prompt.txt"
    prompt_file.write_text(judge_prompt, encoding="utf-8")

    # Call LLM
    call_llm = state["call_llm"]
    raw = call_llm(
        judge_prompt,
        sys_judge_prompt,
        log_path=state["log_path"],
        call_type="judge_optimization",
        round_idx=state["round_idx"],
    )

    # Save raw reply
    reply_file = state["io_dir"] / f"{state['round_idx']}_optimization_strategy_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")

    # Extract JSON strategy
    strategy_json = extract_json(raw)

    # Update state
    return {
        **state,
        "judger_strategy": strategy_json,
        "ncu_metrics_block": metrics_block,
        "next_action": "optimize",
    }
