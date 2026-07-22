import numpy as np
import pytest

from fly_sensor2behavior.flight import (
    ExplicitSpikeCursor,
    MultiRateClock,
    PhaseLockedSpikeGenerator,
    SplayedSpikeGenerator,
)


def test_multirate_clock_has_exact_integer_schedule():
    clock = MultiRateClock(
        duration_s=0.010,
        base_dt_s=0.0001,
        channel_dt_s={"neural": 0.005, "log": 0.001},
    )
    ticks = list(clock)
    assert len(ticks) == 100
    assert sum("neural" in tick.due for tick in ticks) == 2
    assert sum("log" in tick.due for tick in ticks) == 10
    assert ticks[-1].time_s == pytest.approx(0.0099)


def test_multirate_clock_rejects_fractional_ratio():
    with pytest.raises(ValueError, match="integer multiple"):
        MultiRateClock(0.01, 0.0001, {"invalid": 0.00015})


def test_splayed_generator_distributes_low_rate_units():
    generator = SplayedSpikeGenerator(unit_count=6)
    events = []
    for _ in range(2000):
        events.extend(generator.step(rate_hz=5.0, dt_s=0.0001))
    # Six 5 Hz units produce six events over 0.2 s, spread across units rather
    # than one event on every ~200 Hz wingbeat.
    assert len(events) == 6
    assert set(events) == set(range(6))


def test_explicit_cursor_preserves_half_open_event_times():
    cursor = ExplicitSpikeCursor((0.0, 0.001, 0.0015))
    assert cursor.events(0.0, 0.001) == (0.0,)
    assert cursor.events(0.001, 0.002) == (0.001, 0.0015)


def test_phase_locked_generator_emits_only_at_preferred_phase():
    generator = PhaseLockedSpikeGenerator(preferred_phase_rad=np.pi / 3.0)
    phases = []
    phase = 0.0
    dt = 0.0001
    omega = 2.0 * np.pi * 200.0
    for _ in range(200):
        next_phase = phase + omega * dt
        phases.extend(generator.step(100.0, dt, phase, next_phase))
        phase = next_phase
    assert phases
    assert np.allclose(phases, np.pi / 3.0)
