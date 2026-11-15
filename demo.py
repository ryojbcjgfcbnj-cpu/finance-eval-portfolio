"""Synthetic evidence-verification demo using only the Python standard library."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TemporaryFailure(Exception):
    status: int


def with_fallback(primary, backup, *, wait, retries=3):
    events = []
    for attempt in range(retries + 1):
        try:
            value = primary()
            return value, "synthetic-primary", events
        except TemporaryFailure as error:
            if error.status not in {408, 429} and not 500 <= error.status <= 599:
                raise
            delay = 0.25 * 2**attempt if attempt < retries else None
            events.append({"attempt": attempt + 1, "status": error.status, "retry_delay_s": delay})
            if delay is not None:
                wait(delay)
    events.append({"event": "fallback", "to": "synthetic-backup"})
    return backup(), "synthetic-backup", events


def context_check(text, budget):
    size = len(text.encode("utf-8"))
    return {"input_bytes": size, "budget_bytes": budget,
            "state": "ready" if size <= budget else "context_overflow"}


def verify(record):
    observed = record["observed"]
    claimed = record["claimed"]
    if observed is None:
        return "U"
    return "S" if observed == claimed else "C"


def build_report():
    records = [
        {"id": "synthetic-supported", "observed": 12, "claimed": 12},
        {"id": "synthetic-conflict", "observed": 12, "claimed": 15},
        {"id": "synthetic-missing", "observed": None, "claimed": 15},
    ]
    attempts = iter([429, 503, 503, 503])

    def unstable():
        raise TemporaryFailure(next(attempts))

    waits = []
    value, source, events = with_fallback(unstable, lambda: verify(records[1]), wait=waits.append)
    return {
        "schema_version": 1,
        "scope": "synthetic offline demonstration",
        "cases": [{"case_id": row["id"], "verdict": verify(row)} for row in records],
        "resilience": {"judge_model": source, "verdict": value, "events": events, "planned_waits_s": waits},
        "context": context_check("synthetic evidence\n" * 600, 4096),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
