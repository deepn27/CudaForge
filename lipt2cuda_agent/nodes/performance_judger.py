"""Performance judger node for LIPT2CudaAgent."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from lipt2cuda_agent.state import LIPT2CudaAgentState
from prompts.judger_optimization import build_judger_optimization_prompts
from utils.kernel_io import extract_json, extract_cuda_kernel_names
from run_ncu import profile_bench, load_ncu_metrics, metrics_to_prompt


def _write_bench_script(ref_py: Path, test_kernel_py: Path, out_py: Path) -> Path:
    """Generate a self-contained bench script that NCU can profile.

    The script imports Model + get_inputs from the reference file and
    ModelNew from the test kernel, aligns parameters, and runs inference
    so that custom CUDA kernels are actually launched.
    """
    script = dedent(f"""\
        import sys, importlib.util, torch, torch.nn as nn

        def _import(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        ref_mod  = _import("{ref_py}", "ref_mod")
        test_mod = _import("{test_kernel_py}", "test_mod")

        Model      = ref_mod.Model
        ModelNew   = test_mod.ModelNew
        get_inputs = ref_mod.get_inputs

        get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
        init_args = list(get_init_inputs()) if callable(get_init_inputs) else []

        ref_model  = Model(*init_args).cuda().eval()
        test_model = ModelNew(*init_args).cuda().eval()

        # Align parameters from reference to test model
        ref_sd = ref_model.state_dict()
        test_sd = test_model.state_dict()
        if ref_sd and test_sd:
            mapped = {{}}
            ref_keys = list(ref_sd.keys())
            test_keys = list(test_sd.keys())
            for rk, tk in zip(ref_keys, test_keys):
                if ref_sd[rk].shape == test_sd[tk].shape:
                    mapped[tk] = ref_sd[rk]
            if mapped:
                test_model.load_state_dict(mapped, strict=False)

        repeat = 20
        if "--repeat" in sys.argv:
            idx = sys.argv.index("--repeat")
            if idx + 1 < len(sys.argv):
                repeat = int(sys.argv[idx + 1])

        inp = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_inputs()]
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                test_model(*inp)
            torch.cuda.synchronize()
            # Profiled runs
            for _ in range(repeat):
                test_model(*inp)
            torch.cuda.synchronize()
    """)
    out_py.write_text(script, encoding="utf-8")
    return out_py


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

    # Generate a bench runner script that imports the reference model
    # (Model + get_inputs) and the test kernel (ModelNew), then actually
    # runs inference so NCU can capture CUDA kernel launches.
    root_dir = test_kernel_path.parent
    ref_py = root_dir / f"ref_{args.subproc_id}.py"
    bench_py = root_dir / f"bench_ref_inputs_{args.subproc_id}.py"
    _write_bench_script(ref_py, test_kernel_path, bench_py)

    csv_path = profile_bench(
        bench_py=str(bench_py),
        out_csv=f"ncu_temp_{args.subproc_id}.csv",
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
