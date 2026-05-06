"""Decorators used across the package.

Currently exposes a single ``@timed`` decorator that logs the wall-clock
execution time of the wrapped function to stderr. Used to instrument the
profiler and the LLM call so the operator can see where time is spent
during a run. Output goes to stderr so it does not pollute the report
JSON when callers redirect stdout.
"""

from __future__ import annotations

import functools
import sys
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Logs the wall-clock execution time of the wrapped function to stderr."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timed] {func.__name__} ran in {elapsed:.3f}s", file=sys.stderr)
        return result

    return wrapper  # type: ignore[return-value]
