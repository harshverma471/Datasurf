"""The baseline. Written before the first model call, not after the optimizations.

This file exists this early for one reason (plans/modules.md §6): the whole token story is
"naive cost X, optimized cost Y", and a ledger that arrives after the optimizations has no X.
Reconstructing a baseline from memory is the same as not having one.

One `Trace` per answered question. Every node opens a span; the span records what went in,
what came out, how long it took and what it cost, then appends one JSON line to
`data/traces/<run_id>.jsonl`. M4.4's replay reads those files back.

Deliberately *not* recorded: result rows. The app promises uploaded data never reaches disk,
and a trace full of cells would break that promise. Shapes, column names and the SQL are
enough to see what happened. The answer *sentence* is recorded, which can carry a figure —
but only one already shown to the user, never a cell the query merely read. M4.4's replay
will need the rows and can record them under an explicit opt-in when it arrives.
"""

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import config


@dataclass
class Usage:
    """What one model call cost. `usd` is OpenRouter's own figure, not ours.

    There is no price table in this repo on purpose: a hard-coded $/M drifts the moment a
    model is repriced, and OpenRouter returns the real charge per call when asked. `None`
    means the endpoint reported none — true of some `:free` endpoints — which the ledger
    shows as "—" rather than as zero.
    """
    tok_in: int = 0
    tok_out: int = 0
    cached: int = 0
    usd: float | None = None

    def __add__(self, other: "Usage") -> "Usage":
        usd = None if self.usd is None and other.usd is None else (self.usd or 0) + (other.usd or 0)
        return Usage(self.tok_in + other.tok_in, self.tok_out + other.tok_out,
                     self.cached + other.cached, usd)

    @property
    def billed_in(self) -> int:
        """Prompt tokens charged at full rate. Cached reads bill at a tenth on luna."""
        return max(self.tok_in - self.cached, 0)


@dataclass
class Event:
    node: str
    inputs: dict
    out: dict = field(default_factory=dict)
    usage: Usage | None = None
    ms: int = 0
    error: str | None = None
    seq: int = 0


@dataclass
class Trace:
    run_id: str
    question: str = ""
    events: list[Event] = field(default_factory=list)

    @classmethod
    def new(cls, question: str = "") -> "Trace":
        return cls(f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}", question)

    @contextmanager
    def span(self, node: str, **inputs):
        """Time a node, record it, and re-raise. A failed node is a recorded node."""
        event = Event(node=node, inputs=inputs, seq=len(self.events) + 1)
        started = time.perf_counter()
        try:
            yield event
        except Exception as exc:
            event.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event.ms = round((time.perf_counter() - started) * 1000)
            self.events.append(event)
            self._append(event)

    @property
    def usage(self) -> Usage:
        return sum((e.usage for e in self.events if e.usage), Usage())

    @property
    def calls(self) -> int:
        """Model calls, which is not the same as nodes — deterministic nodes spend nothing."""
        return sum(1 for e in self.events if e.usage)

    @property
    def ms(self) -> int:
        return sum(e.ms for e in self.events)

    @property
    def path(self):
        return config.TRACE_DIR / f"{self.run_id}.jsonl"

    def _append(self, event: Event) -> None:
        """Best-effort: a trace that cannot be written must not break the answer it observes."""
        try:
            config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"run": self.run_id, "question": self.question,
                                     **asdict(event)}, default=str) + "\n")
        except OSError:
            pass
