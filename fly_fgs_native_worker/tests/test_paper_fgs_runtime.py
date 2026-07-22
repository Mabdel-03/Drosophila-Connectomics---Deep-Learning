import numpy as np

from fly_sensor2behavior.paper_fgs_runtime import NodePaperFGSCircuitRuntime


def test_paper_runtime_uses_400_hz_explicit_stimulus_and_restricted_motor_cut():
    runtime = NodePaperFGSCircuitRuntime(expected_node_version="v22.22.1")
    try:
        first = runtime.initialize(
            "R83_Fig3a_0_to_90",
            stimulus_time_s=-0.4,
            include_retinal_input=True,
        )
        second = runtime.advance(-0.3975)
        third = runtime.advance(-0.395)
    finally:
        runtime.close()
    assert first.sample_index == 0
    assert second.measurement_time_s == 0.0025
    assert third.measurement_time_s == 0.005
    assert len(first.retinal_input_luminance) == 1441
    assert first.full_cell_state_eligible_motor_input is False
    bridge = third.to_bridge_sample(1)
    assert bridge.sample_index == 1
    assert bridge.measurement_time_s == 0.005
    assert tuple(bridge.nod1_voltage_v) == tuple(first.nod1_voltage_v)
    assert np.isclose(
        bridge.last_control.figure_world_azimuth_rad,
        np.deg2rad(third.stimulus["figure_angle_realized_deg"]),
    )


def test_paper_runtime_800_hz_convergence_mode_preserves_5_ms_motor_cut():
    runtime = NodePaperFGSCircuitRuntime(circuit_dt_s=0.00125)
    try:
        sample = runtime.initialize(
            "R83_Fig3a_0_to_90", stimulus_time_s=-0.4
        )
        for index in range(1, 5):
            sample = runtime.advance(-0.4 + index * 0.00125)
    finally:
        runtime.close()
    assert sample.sample_index == 4
    assert sample.measurement_time_s == 0.005
    assert sample.to_bridge_sample(1).sample_index == 1
