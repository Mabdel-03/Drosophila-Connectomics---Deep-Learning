"""Integer-ratio clocks for deterministic multi-rate simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Tuple


@dataclass(frozen=True)
class ClockTick:
    index: int
    time_s: float
    due: Tuple[str, ...]


class MultiRateClock:
    """Schedule named clocks as integer multiples of a physics timestep.

    Integer ratios eliminate accumulated floating-point scheduling drift.  The
    base physics clock is always due.  A tick denotes the half-open integration
    interval ``[time_s, time_s + base_dt_s)``.
    """

    def __init__(
        self,
        duration_s: float,
        base_dt_s: float,
        channel_dt_s: Mapping[str, float],
        tolerance: float = 1e-9,
    ) -> None:
        if duration_s <= 0.0 or base_dt_s <= 0.0:
            raise ValueError("duration and base timestep must be positive")
        step_count = int(round(duration_s / base_dt_s))
        if abs(step_count * base_dt_s - duration_s) > tolerance * duration_s:
            raise ValueError("duration must be an integer multiple of base_dt_s")
        ratios: Dict[str, int] = {"physics": 1}
        for name, dt_s in channel_dt_s.items():
            if name == "physics":
                raise ValueError("'physics' is a reserved channel name")
            if not name or dt_s < base_dt_s:
                raise ValueError("channel timesteps must be named and >= base_dt_s")
            ratio = int(round(dt_s / base_dt_s))
            if abs(ratio * base_dt_s - dt_s) > tolerance * dt_s:
                raise ValueError(
                    "channel timestep %s must be an integer multiple of base_dt_s" % name
                )
            ratios[name] = ratio
        self.duration_s = float(duration_s)
        self.base_dt_s = float(base_dt_s)
        self.step_count = step_count
        self._ratios = ratios

    @property
    def ratios(self) -> Mapping[str, int]:
        return dict(self._ratios)

    def __iter__(self) -> Iterator[ClockTick]:
        for index in range(self.step_count):
            due = tuple(
                name for name, ratio in self._ratios.items() if index % ratio == 0
            )
            yield ClockTick(index, index * self.base_dt_s, due)
