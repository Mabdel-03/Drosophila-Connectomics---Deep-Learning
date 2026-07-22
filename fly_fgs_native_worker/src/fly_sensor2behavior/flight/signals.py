"""Deterministic conversion of motor signals into spike events."""

from __future__ import annotations

from bisect import bisect_left
from typing import List, Sequence, Tuple

import numpy as np


class ExplicitSpikeCursor:
    """Consume exact event times without changing their scientific status."""

    def __init__(self, spike_times_s: Sequence[float]) -> None:
        self._times = tuple(float(t) for t in spike_times_s)
        self._index = 0

    def events(self, start_s: float, end_s: float) -> Tuple[float, ...]:
        if end_s <= start_s:
            return ()
        # Commands are validated as sorted, but bisect makes reset-less cursors
        # robust when an interval starts after the last one consumed.
        self._index = max(self._index, bisect_left(self._times, start_s))
        first = self._index
        while self._index < len(self._times) and self._times[self._index] < end_s:
            self._index += 1
        return self._times[first : self._index]


class SplayedSpikeGenerator:
    """Low-rate asynchronous motor units with deterministic phase splay.

    This is used for DLM/DVM-like power motor pools.  It deliberately does not
    phase-lock spikes to wingbeats: several low-rate units jointly maintain a
    slowly varying calcium state in asynchronous flight muscle.
    """

    def __init__(self, unit_count: int, initial_offset: float = 0.5) -> None:
        if unit_count < 1:
            raise ValueError("unit_count must be positive")
        self.unit_count = int(unit_count)
        self._phase = (
            np.arange(self.unit_count, dtype=float) + float(initial_offset)
        ) / self.unit_count

    def step(self, rate_hz: float, dt_s: float) -> Tuple[int, ...]:
        if rate_hz < 0.0 or dt_s <= 0.0:
            raise ValueError("rate must be non-negative and dt positive")
        old = self._phase.copy()
        self._phase += rate_hz * dt_s
        crossings = np.floor(self._phase).astype(int) - np.floor(old).astype(int)
        emitted: List[int] = []
        for unit, count in enumerate(crossings):
            emitted.extend([unit] * int(count))
        self._phase %= 1.0
        return tuple(emitted)


class PhaseLockedSpikeGenerator:
    """Infer at most one steering spike per wingbeat at a named phase.

    A sigma-delta credit accumulator preserves the requested long-run rate
    without random sampling.  These events must retain ``inferred_rate``
    provenance; the generator is not a substitute for measured MN spike time.
    """

    def __init__(self, preferred_phase_rad: float) -> None:
        self.preferred_phase_rad = float(preferred_phase_rad) % (2.0 * np.pi)
        self._credit = 0.0

    def step(
        self,
        rate_hz: float,
        dt_s: float,
        phase_start_rad: float,
        phase_end_rad: float,
    ) -> Tuple[float, ...]:
        if rate_hz < 0.0 or dt_s <= 0.0 or phase_end_rad < phase_start_rad:
            raise ValueError("invalid phase-locked generator step")
        self._credit += rate_hz * dt_s
        two_pi = 2.0 * np.pi
        first_cycle = int(np.ceil((phase_start_rad - self.preferred_phase_rad) / two_pi))
        last_cycle = int(np.floor((phase_end_rad - self.preferred_phase_rad) / two_pi))
        events: List[float] = []
        for cycle in range(first_cycle, last_cycle + 1):
            crossing = self.preferred_phase_rad + cycle * two_pi
            if phase_start_rad <= crossing < phase_end_rad and self._credit >= 1.0:
                self._credit -= 1.0
                events.append(self.preferred_phase_rad)
        # Avoid an unbounded reservoir if the requested rate exceeds one event
        # per wingbeat. Synchronous steering muscles cannot realize that request
        # under this first-order model, so excess credit is discarded.
        self._credit = min(self._credit, 1.0)
        return tuple(events)
