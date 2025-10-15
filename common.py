from typing import Any
import json
import os


def canonicalize(obj: Any) -> Any:
    """
    Produce a JSON-serializable structure with stable ordering for dict keys and
    normalized simple types. Lists keep order; dicts are sorted by key.
    Tuples converted to lists. Removes keys with value None to reduce cache misses
    caused by implicit defaults.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        # Drop None values to avoid mismatches when some calls omit default keys
        filtered: dict[str, Any] = {
            str(k): v for k, v in obj.items() if v is not None}  # type: ignore
        return {k: canonicalize(filtered[k]) for k in sorted(filtered)}
    if isinstance(obj, (list, tuple)):
        items: list[Any] = list(obj)  # type: ignore[arg-type]
        return [canonicalize(i) for i in items]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # Fallback: try to get dict-like view
    if hasattr(obj, "dict"):
        return canonicalize(obj.dict())  # type: ignore
    if hasattr(obj, "model_dump"):
        return canonicalize(obj.model_dump())  # type: ignore
    return str(obj)


def canonical_json(payload: Any) -> str:
    canonical = canonicalize(payload)
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def calculate_optimal_workers(task_type: str = "io_heavy", cpu_multiplier: float = 30.0) -> int:
    """
    Calculate optimal thread count for different task types.

    Args:
        task_type: Type of workload
            - "io_heavy": Network I/O like OpenAI API calls (default)
            - "cpu_bound": CPU-intensive tasks
            - "mixed": Mix of I/O and CPU work
        cpu_multiplier: Multiplier for I/O-bound tasks (default: 30x CPU count)

    Returns:
        Recommended number of worker threads

    Formula rationale:
        - CPU-bound: 1x CPU count (avoid context switching overhead)
        - I/O-bound: 10-50x CPU count (threads spend time waiting)
        - OpenAI API: 20-100+ concurrent requests typically work well
    """
    cpu_count = os.cpu_count() or 4  # Fallback to 4 if detection fails

    if task_type == "cpu_bound":
        return cpu_count
    elif task_type == "mixed":
        return min(cpu_count * 4, 20)  # Conservative for mixed workloads
    else:  # io_heavy (default)
        optimal = int(cpu_count * cpu_multiplier)
        # Reasonable bounds: at least 10, at most 100 for OpenAI API
        return max(10, min(optimal, 100))
