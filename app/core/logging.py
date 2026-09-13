"""Logging setup + a millisecond stopwatch used by the performance panel."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

_CONFIGURED = False


def setup_logging(debug: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    # uvicorn access logs are noisy for a polling UI
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass
class Timings:
    """Accumulates named phase durations in milliseconds.

    Every op execution builds one of these; it is returned verbatim to the
    performance panel so the user can see decode / process / encode split.
    """

    marks: dict[str, float] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            dur = (time.perf_counter() - start) * 1000.0
            self.marks[name] = round(self.marks.get(name, 0.0) + dur, 3)

    def add(self, name: str, ms: float) -> None:
        self.marks[name] = round(self.marks.get(name, 0.0) + ms, 3)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 3)

    def as_dict(self) -> dict[str, float]:
        out = dict(self.marks)
        out["total"] = self.total_ms
        return out
