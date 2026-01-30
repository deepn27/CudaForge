"""Checkpointing configuration for LIPT2CudaAgent."""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from scripts.individual import KernelIndividual


class LIPT2CudaAgentSerializer:
    """Custom serializer for LIPT2CudaAgent state objects."""

    @staticmethod
    def serialize_kernel_individual(obj: KernelIndividual) -> dict:
        """Serialize KernelIndividual to dict."""
        return {
            "__type__": "KernelIndividual",
            "id": obj.id,
            "code": obj.code,
            "metrics": obj.metrics,
            "score": obj.score,
            "feedback": obj.feedback,
            "code_path": str(obj.code_path) if obj.code_path else None,
        }

    @staticmethod
    def deserialize_kernel_individual(data: dict) -> KernelIndividual:
        """Deserialize dict to KernelIndividual."""
        ind = KernelIndividual(data["code"])
        ind.id = data["id"]
        ind.metrics = data["metrics"]
        ind.score = data["score"]
        ind.feedback = data["feedback"]
        ind.code_path = Path(data["code_path"]) if data["code_path"] else None

        # Update global ID counter to avoid collisions
        if ind.id >= KernelIndividual._next_id:
            KernelIndividual._next_id = ind.id + 1

        return ind

    @staticmethod
    def serialize_path(obj: Path) -> dict:
        """Serialize Path to dict."""
        return {
            "__type__": "Path",
            "path": str(obj),
        }

    @staticmethod
    def deserialize_path(data: dict) -> Path:
        """Deserialize dict to Path."""
        return Path(data["path"])

    @staticmethod
    def serialize_namespace(obj: argparse.Namespace) -> dict:
        """Serialize argparse.Namespace to dict."""
        return {
            "__type__": "Namespace",
            "vars": vars(obj),
        }

    @staticmethod
    def deserialize_namespace(data: dict) -> argparse.Namespace:
        """Deserialize dict to argparse.Namespace."""
        ns = argparse.Namespace()
        for key, value in data["vars"].items():
            setattr(ns, key, value)
        return ns

    @classmethod
    def dumps(cls, obj: Any) -> bytes:
        """Serialize object to bytes with custom handlers."""
        def default_serializer(o):
            if isinstance(o, KernelIndividual):
                return cls.serialize_kernel_individual(o)
            elif isinstance(o, Path):
                return cls.serialize_path(o)
            elif isinstance(o, argparse.Namespace):
                return cls.serialize_namespace(o)
            raise TypeError(f"Object of type {type(o)} is not serializable")

        # Use pickle with custom protocol
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def loads(cls, data: bytes) -> Any:
        """Deserialize bytes to object with custom handlers."""
        return pickle.loads(data)


def get_checkpoint_path(task_name: str, base_dir: Path = Path(".lipt2cuda")) -> Path:
    """Get checkpoint database path for a task.

    Args:
        task_name: Name of the task
        base_dir: Base directory for checkpoints

    Returns:
        Path to SQLite checkpoint file
    """
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = checkpoint_dir / f"{task_name}_{timestamp}.db"

    return db_path


def get_thread_id(task_name: str, batch_timestamp: str, round_idx: int) -> str:
    """Generate thread ID for checkpointing.

    Args:
        task_name: Name of the task (without extension)
        batch_timestamp: Timestamp when the batch started
        round_idx: Current round index

    Returns:
        Thread ID string
    """
    return f"{task_name}_{batch_timestamp}_{round_idx:03d}"


def create_checkpoint_saver(db_path: Path) -> MemorySaver:
    """Create and initialize MemorySaver (in-memory checkpointing).

    Args:
        db_path: Path to SQLite database file (ignored for MemorySaver)

    Returns:
        Initialized MemorySaver
    """
    # For now, use MemorySaver instead of AsyncSqliteSaver
    # TODO: Add AsyncSqliteSaver support when langgraph-checkpoint-sqlite is available
    return MemorySaver()
