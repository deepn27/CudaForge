"""LangGraph construction for LIPT2CudaAgent workflow."""
from __future__ import annotations
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from lipt2cuda_agent.state import LIPT2CudaAgentState
from lipt2cuda_agent.nodes.seed_generator import seed_generator_node
from lipt2cuda_agent.nodes.benchmark import benchmark_node
from lipt2cuda_agent.nodes.correctness_judger import correctness_judger_node
from lipt2cuda_agent.nodes.repair import repair_node
from lipt2cuda_agent.nodes.performance_judger import performance_judger_node
from lipt2cuda_agent.nodes.optimization import optimization_node
from lipt2cuda_agent.routers import (
    route_after_benchmark,
    route_repair_or_optimize,
    update_state_node,
)


def build_lipt2cuda_graph(
    checkpointer: Optional[MemorySaver] = None
) -> StateGraph:
    """Build the LIPT2CudaAgent LangGraph workflow.

    Graph Structure:
        START → seed_generator → benchmark → route_after_benchmark
            ↓ (if max_rounds not reached)
        update_state → route_repair_or_optimize
            ↓ (if not runnable)        ↓ (if runnable)
        correctness_judger      performance_judger
            ↓                           ↓
        repair                      optimization
            ↓                           ↓
        benchmark ────────────────────→ benchmark
            ↓
        (loop back to route_after_benchmark)

    Args:
        checkpointer: Optional AsyncSqliteSaver for state persistence

    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(LIPT2CudaAgentState)

    # Add nodes
    workflow.add_node("seed_generator", seed_generator_node)
    workflow.add_node("benchmark", benchmark_node)
    workflow.add_node("update_state", update_state_node)
    workflow.add_node("correctness_judger", correctness_judger_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("performance_judger", performance_judger_node)
    workflow.add_node("optimization", optimization_node)

    # Set entry point
    workflow.set_entry_point("seed_generator")

    # Add edges
    # Seed path: seed_generator → benchmark
    workflow.add_edge("seed_generator", "benchmark")

    # After benchmark: check if max rounds reached
    workflow.add_conditional_edges(
        "benchmark",
        route_after_benchmark,
        {
            "update_state": "update_state",
            "END": END,
        },
    )

    # After state update: route to repair or optimize
    workflow.add_conditional_edges(
        "update_state",
        route_repair_or_optimize,
        {
            "correctness_judger": "correctness_judger",
            "performance_judger": "performance_judger",
        },
    )

    # Repair path: correctness_judger → repair → benchmark
    workflow.add_edge("correctness_judger", "repair")
    workflow.add_edge("repair", "benchmark")

    # Optimize path: performance_judger → optimization → benchmark
    workflow.add_edge("performance_judger", "optimization")
    workflow.add_edge("optimization", "benchmark")

    # Compile graph
    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()

    return app
