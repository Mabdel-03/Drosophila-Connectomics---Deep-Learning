import numpy as np

from fly_sensor2behavior.flight import (
    FlightEpisodeRunner,
    FlightSimulationConfig,
    MuscleClass,
    default_motor_commands,
)
from fly_sensor2behavior.flight.bridge import CausalNeuralBridge
from fly_sensor2behavior.nod1 import legacy_result_to_circuit_trace


NOD1_LEFT = ("720575940628438427", "720575940625528556")
NOD1_RIGHT = ("720575940623997949", "720575940629456860")


def _legacy_result(duration_s=0.100, dt_s=0.005):
    count = int(round(duration_s / dt_s)) + 1
    times = [index * dt_s for index in range(count)]
    voltage = {}
    for root_id in NOD1_LEFT:
        voltage[root_id] = [-50.0] * count
    for root_id in NOD1_RIGHT:
        voltage[root_id] = [-60.0] * count
    return {
        "time": times,
        "voltage": voltage,
        "steering": [1.0] * count,
        "meta": {"logicVersion": "legacy-nod1-v0.5.0"},
    }


def test_legacy_import_converts_mv_and_forbids_population_mean_motor_use():
    trace = legacy_result_to_circuit_trace(_legacy_result())
    assert len(trace.signals) == 4
    assert all(signal.unit == "V" for signal in trace.signals)
    assert all(signal.neuron.entity_id.isdigit() for signal in trace.signals)
    left = [
        signal
        for signal in trace.signals
        if signal.neuron.entity_id in NOD1_LEFT
    ]
    assert all(signal.values[0] == -0.05 for signal in left)
    assert "steering" in trace.provenance.filters["excluded_motor_fields"]
    assert "population_mean_activity" in trace.provenance.notes


def test_frozen_visual_trace_runs_through_causal_motor_and_individual_muscles():
    circuit = legacy_result_to_circuit_trace(_legacy_result())
    bridge = CausalNeuralBridge().run(circuit, seed=41)
    descending, motor = bridge.descending, bridge.wing_motor
    assert descending.channel("DNp26", "right").samples[-1].rate_hz > 0.0
    assert descending.channel("DNp26", "left").samples[-1].rate_hz == 0.0

    baseline_flight_state = tuple(
        command
        for command in default_motor_commands()
        if command.muscle_class is not MuscleClass.STEERING
    )
    commands = baseline_flight_state + motor.to_flight_motor_commands()
    result = FlightEpisodeRunner().run(
        FlightSimulationConfig(
            duration_s=0.100,
            physics_dt_s=0.0001,
            neural_dt_s=0.005,
            logging_dt_s=0.001,
            seed=41,
            motor_commands=commands,
        )
    )

    assert result.diagnostics.inferred_spike_count > 0
    assert result.diagnostics.exact_spike_count == 0
    assert np.max(result.muscle_force_n["right:iv2"]) > 0.0
    assert np.max(result.muscle_force_n["left:iv2"]) == 0.0
    assert result.diagnostics.metrics["neural_update_count"] == 20.0
    assert np.all(np.isfinite(result.position_world_m))
