# LIPT2CudaAgent: LangGraph Migration

This directory contains the LangGraph-based implementation of the CUDA kernel optimization workflow, renamed from CudaForge to LIPT2CudaAgent.

## Overview

LIPT2CudaAgent uses LangGraph to orchestrate a multi-agent workflow for generating, benchmarking, and optimizing CUDA kernels. The workflow includes:

1. **Seed Generation**: Generate initial CUDA kernels from task specifications
2. **Benchmarking**: Compile and benchmark kernels with subprocess isolation
3. **Correctness Analysis**: Analyze compilation/runtime errors
4. **Repair**: Fix broken kernels based on error analysis
5. **Performance Analysis**: Profile with NVIDIA Nsight Compute (NCU)
6. **Optimization**: Improve runnable kernels based on performance metrics

## Architecture

### State Schema

The workflow state (`LIPT2CudaAgentState`) tracks:
- Current kernel and metrics
- Best kernel and score
- Round history
- Phase control
- LLM callbacks
- Directory paths

### Graph Flow

```
START → seed_generator → benchmark → route_after_benchmark
    ↓ (if max_rounds not reached)
update_state → route_repair_or_optimize
    ↓ (not runnable)        ↓ (runnable)
correctness_judger      performance_judger
    ↓                           ↓
repair                      optimization
    ↓                           ↓
benchmark ←───────────────────→ benchmark
    ↓
(loop back)
```

### Nodes

1. **SeedGeneratorNode** (`langgraph/nodes/seed_generator.py`)
   - Generates initial kernel (Round 0)
   - Uses `build_seed_prompt` from prompts

2. **BenchmarkNode** (`langgraph/nodes/benchmark.py`)
   - Compiles and benchmarks kernels
   - Subprocess isolation for CUDA context
   - Updates metrics and scores

3. **CorrectnessJudgerNode** (`langgraph/nodes/correctness_judger.py`)
   - Analyzes compilation/runtime errors
   - Generates JSON repair strategy

4. **RepairNode** (`langgraph/nodes/repair.py`)
   - Fixes broken kernels
   - Uses error analysis from judger

5. **PerformanceJudgerNode** (`langgraph/nodes/performance_judger.py`)
   - Runs NCU profiling
   - Analyzes metrics
   - Generates JSON optimization strategy

6. **OptimizationNode** (`langgraph/nodes/optimization.py`)
   - Optimizes runnable kernels
   - Applies performance improvements

### Routers

- **route_after_benchmark**: Check if max rounds reached
- **route_repair_or_optimize**: Route based on kernel runnable status
- **update_state_node**: Update scores, history, best kernel

## Usage

### Basic Usage

```bash
# Run on a single task
python langgraph_main.py KernelBench/level1/1_Square_matrix_multiplication_.py \
    --round 10 \
    --gpu "Quadro RTX 6000"

# Run with checkpointing
python langgraph_main.py KernelBench/level1/1_Square_matrix_multiplication_.py \
    --round 10 \
    --enable_checkpointing \
    --checkpoint_dir .lipt2cuda

# Run with Anthropic reasoning models (Claude)
python langgraph_main.py KernelBench/level1/1_Square_matrix_multiplication_.py \
    --server_type anthropic \
    --model_name claude-opus-4-5 \
    --is_reasoning_model \
    --budget_tokens 15000 \
    --round 10
```

### Checkpointing

Enable SQLite checkpointing for:
- Stop/resume workflows
- State inspection
- Debugging

Checkpoints are stored in:
```
.lipt2cuda/checkpoints/{task_name}_{timestamp}.db
```

### CLI Arguments

All arguments from `main.py` are supported, plus:

- `--enable_checkpointing`: Enable SQLite checkpointing
- `--checkpoint_dir`: Directory for checkpoint databases (default: `.lipt2cuda`)
- `--budget_tokens`: Budget tokens for Anthropic reasoning models (default: 10000)
- `--is_reasoning_model`: Enable reasoning mode for Anthropic models (flag)

## File Structure

```
CudaForge/
├── langgraph/
│   ├── __init__.py
│   ├── state.py                    # State schema
│   ├── graph.py                    # Graph construction
│   ├── routers.py                  # Routing logic
│   ├── checkpointing.py            # Checkpoint config
│   └── nodes/
│       ├── __init__.py
│       ├── seed_generator.py
│       ├── benchmark.py
│       ├── correctness_judger.py
│       ├── repair.py
│       ├── performance_judger.py
│       └── optimization.py
├── langgraph_main.py               # Entry point
├── main.py                         # Original (for comparison)
└── LANGGRAPH_README.md             # This file
```

## Dependencies

Added to `environment.yml`:

```yaml
- aiosqlite==0.20.0
- langchain-core==0.3.30
- langgraph==0.2.50
```

Install with:

```bash
conda env update -f environment.yml
```

## Differences from Original

### Advantages

1. **Modular**: Each node is a separate, testable function
2. **Stateful**: Built-in state management and checkpointing
3. **Observable**: Native streaming and state inspection
4. **Resumable**: SQLite checkpointing for long workflows
5. **Maintainable**: Clear separation of concerns

### Compatibility

- Output structure identical to original
- Same prompt templates
- Same benchmarking logic
- Same LLM interface

## Migration Status

✅ Phase 1: Foundation (Complete)
- Directory structure
- State schema
- Dependencies

✅ Phase 2: Node Implementation (Complete)
- All 6 nodes implemented
- Subprocess isolation preserved

✅ Phase 3: Graph Construction (Complete)
- Routing logic
- State updates

✅ Phase 4: Integration (Complete)
- Entry point
- Checkpointing
- CLI arguments

⏸️ Phase 5: Testing & Validation (Pending)
- Single-task comparison test
- Output equivalence verification
- Checkpointing test

⏸️ Phase 6: Documentation (In Progress)
- This README
- Migration guide (pending)
- API documentation (pending)

## Future Enhancements

1. **LangSmith Integration**: Add observability with environment variables
2. **Automated Tests**: Unit tests for nodes and integration tests
3. **Performance Benchmarking**: Compare execution time vs original
4. **Advanced Checkpointing**: Resume from specific rounds

## Troubleshooting

### Import Errors

Make sure you're running from the repository root:

```bash
cd /path/to/CudaForge
python langgraph_main.py ...
```

### Checkpoint Issues

Delete old checkpoints if you encounter state errors:

```bash
rm -rf .lipt2cuda/checkpoints/*.db
```

### CUDA Context Errors

The benchmark node uses subprocess isolation to prevent CUDA context issues. If you still encounter problems:

1. Check CUDA device availability
2. Verify `--device` argument matches your GPU
3. Check memory usage with `nvidia-smi`

## Contributing

When adding new features:

1. Follow the existing node structure
2. Update state schema if needed
3. Add routing logic for new paths
4. Document changes in this README

## License

Same as parent project.
