from dataclasses import replace

import numpy as np

from fly_sensor2behavior.flight.paper_open_loop import (
    PaperOpenLoopConfig,
    PaperOpenLoopSimulator,
)
from fly_sensor2behavior.flight.rigid_body import RigidBodyState
from fly_sensor2behavior.fly_fgs import FLY_FGS_NOD1_ROOT_IDS
from fly_sensor2behavior.paper_fgs import load_protocol, sample_protocol
from fly_sensor2behavior.paper_fgs_runtime import PaperFGSCircuitSample
from fly_sensor2behavior.paper_tether import paper_torque_from_support_reaction


class _FakeCircuit:
    def __init__(self):
        self.index = -1
        self.protocol = None
        self.ready_receipt = {
            "protocol_version": "1.0.0",
            "assets": {"test": {"sha256": "0" * 64}},
        }

    def _sample(self, stimulus_time_s):
        stimulus = sample_protocol(self.protocol, [stimulus_time_s])
        stimulus = {
            key: value[0].item() if hasattr(value[0], "item") else value[0]
            for key, value in stimulus.items()
        }
        return PaperFGSCircuitSample(
            sample_index=self.index,
            measurement_time_s=self.index * 0.0025,
            availability_time_s=self.index * 0.0025,
            stimulus_time_s=stimulus_time_s,
            stimulus=stimulus,
            nod1_voltage_v={root_id: -0.055 for root_id in FLY_FGS_NOD1_ROOT_IDS},
            pooled_readout={},
        )

    def initialize(self, protocol_id, **kwargs):
        self.protocol = _SHORT_PROTOCOL
        assert protocol_id == self.protocol.protocol_id
        self.index = 0
        return self._sample(kwargs["stimulus_time_s"])

    def advance(self, stimulus_time_s):
        self.index += 1
        return self._sample(stimulus_time_s)

    def close(self):
        pass


class _FakePhysics:
    def __init__(self, output_scale):
        self.output_scale = output_scale
        self.wings = np.zeros(6)
        self.reported = 0.0
        self.state = RigidBodyState(
            position_world_m=np.zeros(3),
            velocity_world_m_s=np.zeros(3),
            quaternion_body_to_world=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity_body_rad_s=np.zeros(3),
        )

    def calibrate_paper_yaw_sign(self):
        return {"passed": True}

    def default_initial_state(self):
        return self.state.copy()

    def reset(self, initial_state):
        self.state = initial_state.copy()
        self.reported = 0.0

    def step(self, wings, _force, _moment, dt_s):
        assert dt_s == 0.0001
        self.wings = np.array(
            [
                wings.stroke_rad[0],
                wings.deviation_rad[0],
                wings.angle_of_attack_rad[0],
                wings.stroke_rad[1],
                wings.deviation_rad[1],
                wings.angle_of_attack_rad[1],
            ]
        )
        self.reported = self.output_scale * float(wings.stroke_rad[0] - wings.stroke_rad[1]) * 1e-9
        return self.state.copy()

    def tether_torque_sample(self):
        return paper_torque_from_support_reaction(
            np.zeros(3), np.array([0.0, 0.0, self.reported])
        )

    def body_state(self):
        return self.state.copy()

    def wing_joint_state(self):
        return self.wings.copy(), np.zeros(6)

    def head_transform_body(self):
        return np.eye(4)

    def yaw_equation_residual_n_m(self):
        return 0.0

    def body_constraint_diagnostics(self):
        return {
            "body_translation_drift_m": 0.0,
            "body_rotation_drift_rad": 0.0,
            "body_yaw_rate_rad_s": 0.0,
        }

    def ground_contact_count(self):
        return 0

    def provenance_metadata(self):
        return {"backend": "fake"}


_SHORT_PROTOCOL = replace(
    load_protocol("R83_Fig3a_0_to_90"),
    trial_duration_s=0.02,
    phase_transition_start_s=0.005,
    phase_transition_duration_s=0.005,
)


def _run(output_scale):
    config = PaperOpenLoopConfig(protocol=_SHORT_PROTOCOL, repetitions=1)
    simulator = PaperOpenLoopSimulator(
        config,
        physics_adapter=_FakePhysics(output_scale),
        circuit_factory=_FakeCircuit,
    )
    return simulator.run()


def test_open_loop_motor_output_cannot_change_the_stimulus():
    low, low_metadata = _run(-100.0)
    high, high_metadata = _run(100.0)
    for name in (
        "figure_angle_command_deg",
        "figure_angle_realized_deg",
        "ground_angle_command_deg",
        "ground_angle_realized_deg",
        "relative_phase_realized_deg",
    ):
        assert np.array_equal(low[name], high[name])
    assert low["time_s"].shape == (20,)
    assert low["raw_physics_time_s"].shape == (200,)
    assert low["head_transform_body"].shape == (1, 20, 4, 4)
    assert low_metadata["trial_receipts"][0]["circuit_runtime"] is not None
    assert high_metadata["config"]["effector_laterality_hypothesis"] == "raw_l_to_physical_right"


def test_wingbeat_phases_are_uniformly_stratified_and_seeded():
    config = PaperOpenLoopConfig(protocol=_SHORT_PROTOCOL, repetitions=100)
    phases = np.asarray(
        [config.wingbeat_phase_radians(trial_id) for trial_id in range(100)]
    )
    expected = 2 * np.pi * (np.arange(100) + 0.5) / 100
    np.testing.assert_allclose(np.sort(phases), expected, rtol=0.0, atol=1e-15)
    assert not np.array_equal(phases, expected)
    fixed = replace(config, wingbeat_phase_mode="fixed")
    assert {fixed.wingbeat_phase_radians(index) for index in range(100)} == {0.0}
