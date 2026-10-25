"""The last N requests, and the totals the dashboard shows. In memory: a restart starts the count again."""

import statistics
from collections import deque

from app.schemas import GateEvent, GateStats


class EventLog:
    def __init__(self, size: int) -> None:
        self._events: deque[GateEvent] = deque(maxlen=size)

    def add(self, event: GateEvent) -> None:
        self._events.append(event)

    def recent(self, limit: int) -> list[GateEvent]:
        return list(self._events)[-limit:][::-1]  # newest first

    def stats(self) -> GateStats:
        events = list(self._events)
        answered = [event for event in events if event.outcome == "answered"]
        decision_times = [event.decision_ms for event in events if event.decision_ms is not None]
        upstream_times = [event.upstream_ms for event in answered if event.upstream_ms is not None]

        def tokens(route: str) -> int:
            return sum(
                (event.input_tokens or 0) + (event.output_tokens or 0) for event in answered if event.route == route
            )

        return GateStats(
            requests=len(events),
            answered=len(answered),
            cached=sum(event.outcome == "cached" for event in events),
            blocked=sum(event.outcome == "blocked" for event in events),
            failed=sum(event.outcome == "failed" for event in events),
            small_routed=sum(event.route == "small" for event in answered),
            strong_routed=sum(event.route == "strong" for event in answered),
            masked_identifiers=sum(event.masked_identifiers for event in events),
            small_tokens=tokens("small"),
            strong_tokens=tokens("strong"),
            median_decision_ms=statistics.median(decision_times) if decision_times else None,
            median_upstream_ms=statistics.median(upstream_times) if upstream_times else None,
        )
