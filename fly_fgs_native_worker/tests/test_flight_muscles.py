import numpy as np

from fly_sensor2behavior.flight import (
    AsynchronousPowerMuscle,
    PhaseCodedSteeringMuscle,
    TensionMuscle,
    ThoraxOscillator,
)


def test_power_spike_changes_slow_calcium_state_not_one_wingbeat():
    stimulated = AsynchronousPowerMuscle()
    control = AsynchronousPowerMuscle()
    dt = 0.0001
    stimulated.step(1, stretch=0.0, shortening_velocity_s=0.0, dt_s=dt)
    control.step(0, stretch=0.0, shortening_velocity_s=0.0, dt_s=dt)
    initial_difference = stimulated.activation - control.activation
    for _ in range(100):  # 10 ms, approximately two wingbeats
        stimulated.step(0, 0.0, 0.0, dt)
        control.step(0, 0.0, 0.0, dt)
    assert stimulated.calcium > control.calcium
    assert stimulated.activation > control.activation
    assert stimulated.activation - control.activation > initial_difference


def test_steering_effect_depends_on_spike_phase():
    preferred = 0.4
    at_preferred = PhaseCodedSteeringMuscle(preferred)
    opposite = PhaseCodedSteeringMuscle(preferred)
    positive = at_preferred.step((preferred,), dt_s=0.0001)
    negative = opposite.step((preferred + np.pi,), dt_s=0.0001)
    assert positive > 0.0
    assert negative < 0.0
    assert at_preferred.activation == opposite.activation


def test_tension_activation_increases_thoracic_resonance():
    tension = TensionMuscle()
    baseline_scale = tension.step(0, 0.0001)
    activated_scale = tension.step(3, 0.0001)
    assert activated_scale > baseline_scale

    control_oscillator = ThoraxOscillator()
    activated_oscillator = ThoraxOscillator()
    for _ in range(100):
        control_oscillator.step(0.62, baseline_scale, 0.0001)
        activated_oscillator.step(0.62, activated_scale, 0.0001)
    assert activated_oscillator.frequency_hz > control_oscillator.frequency_hz
