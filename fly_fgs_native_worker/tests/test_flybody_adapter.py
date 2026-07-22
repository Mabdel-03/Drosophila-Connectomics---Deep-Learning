import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fly_sensor2behavior.flybody_adapter import (
    PINNED_FLYGYM_VERSION,
    WORKER_IMAGE_DIGEST_ENV,
    FlyBodyDependencyError,
    FlyBodyPhysicsAdapter,
    FlyBodyWorkerConfig,
    WingAxisTorqueMap,
    assert_worker_compatible,
    declared_worker_image_digest,
    dependency_versions,
    worker_dependency_lock_path,
    worker_dependency_lock_sha256,
)
from fly_sensor2behavior.flight.types import RigidBodyState, WingKinematics
from fly_sensor2behavior.schema import REVIEWED_FLYBODY_WING_AXIS_ORDER


def test_declared_worker_image_digest_is_optional_and_fail_closed(monkeypatch):
    digest = "sha256:" + ("1a" * 32)
    monkeypatch.delenv(WORKER_IMAGE_DIGEST_ENV, raising=False)
    assert declared_worker_image_digest() is None

    monkeypatch.setenv(WORKER_IMAGE_DIGEST_ENV, digest)
    assert declared_worker_image_digest() == digest

    for invalid in (
        "",
        digest.removeprefix("sha256:"),
        "sha256:" + ("A" * 64),
        "sha256:" + ("0" * 63),
        "md5:" + ("0" * 64),
    ):
        monkeypatch.setenv(WORKER_IMAGE_DIGEST_ENV, invalid)
        with pytest.raises(ValueError, match=WORKER_IMAGE_DIGEST_ENV):
            declared_worker_image_digest()


def test_worker_dependency_lock_receipt_matches_repository_bytes():
    path = worker_dependency_lock_path()
    assert path.name == "requirements.lock"
    assert worker_dependency_lock_sha256() == (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_worker_config_rejects_nonpositive_timestep():
    try:
        FlyBodyWorkerConfig(timestep_s=0.0)
    except ValueError as exc:
        assert "timestep" in str(exc)
    else:
        raise AssertionError("nonpositive timestep was accepted")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("timestep_s", float("nan")),
        ("spawn_height_m", float("inf")),
        ("max_abs_wing_torque_n_m", float("nan")),
    ),
)
def test_worker_config_rejects_nonfinite_values(field, value):
    with pytest.raises(ValueError, match="must be positive"):
        FlyBodyWorkerConfig(**{field: value})


def test_default_initial_state_is_a_defensive_copy_without_worker_dependencies():
    # Exercise the state-cache contract without constructing MuJoCo/FlyGym. A
    # caller may freely mutate its episode state, but must not mutate the exact
    # compiled airborne state retained by the adapter for the next episode.
    adapter = FlyBodyPhysicsAdapter.__new__(FlyBodyPhysicsAdapter)
    cached = RigidBodyState(
        position_world_m=np.array([0.001, -0.002, 0.123]),
        velocity_world_m_s=np.array([0.4, 0.5, -0.6]),
        quaternion_body_to_world=(
            np.array([0.9, 0.1, 0.2, 0.3])
            / np.linalg.norm(np.array([0.9, 0.1, 0.2, 0.3]))
        ),
        angular_velocity_body_rad_s=np.array([1.0, -2.0, 3.0]),
    )
    adapter._default_initial_state = cached

    first = adapter.default_initial_state()
    first.position_world_m[:] = 99.0
    first.velocity_world_m_s[:] = 88.0
    first.quaternion_body_to_world[:] = 77.0
    first.angular_velocity_body_rad_s[:] = 66.0
    second = adapter.default_initial_state()

    assert second is not cached
    np.testing.assert_array_equal(second.position_world_m, [0.001, -0.002, 0.123])
    np.testing.assert_array_equal(second.velocity_world_m_s, [0.4, 0.5, -0.6])
    np.testing.assert_array_equal(
        second.quaternion_body_to_world,
        np.array([0.9, 0.1, 0.2, 0.3])
        / np.linalg.norm(np.array([0.9, 0.1, 0.2, 0.3])),
    )
    np.testing.assert_array_equal(
        second.angular_velocity_body_rad_s, [1.0, -2.0, 3.0]
    )


def test_dependency_probe_is_non_importing_and_fail_closed():
    versions = dependency_versions()
    assert set(versions) == {"flygym", "mujoco"}
    if versions["flygym"] != PINNED_FLYGYM_VERSION:
        try:
            assert_worker_compatible()
        except FlyBodyDependencyError:
            pass
        else:
            raise AssertionError("unpinned worker dependency was accepted")


def test_virtual_hinge_torque_mapping_is_explicit_and_six_axis():
    wings = WingKinematics(
        phase_rad=0.0,
        frequency_hz=200.0,
        stroke_rad=np.zeros(2),
        stroke_velocity_rad_s=np.zeros(2),
        stroke_acceleration_rad_s2=np.zeros(2),
        angle_of_attack_rad=np.zeros(2),
        deviation_rad=np.zeros(2),
        generalized_torque_n_m=np.array([2e-9, -3e-9]),
    )
    mapper = WingAxisTorqueMap(
        left_yaw_roll_pitch=(1.0, 0.5, 0.0),
        right_yaw_roll_pitch=(1.0, 0.0, -0.5),
    )
    assert mapper.map_torque(wings) == pytest.approx(
        (2e-9, 1e-9, 0.0, -3e-9, 0.0, 1.5e-9)
    )


def test_analytic_fallback_is_one_beat_periodic_including_roll_harmonic():
    from fly_sensor2behavior.flybody_adapter import _analytic_wing_pattern

    # Upstream repeats one 0..2*pi lookup-table cycle.  The 1.5-harmonic roll
    # term must therefore restart each beat rather than alternate between two
    # different cycles when evaluated at an unbounded phase.
    phase = 1.234
    angles, derivatives = _analytic_wing_pattern(phase, 0.91, 1.07)
    wrapped_angles, wrapped_derivatives = _analytic_wing_pattern(
        phase + 2.0 * np.pi, 0.91, 1.07
    )

    np.testing.assert_allclose(wrapped_angles, angles, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        wrapped_derivatives, derivatives, rtol=0.0, atol=1.0e-14
    )


def _capture_analytic_controller(
    monkeypatch, *, physics_timestep_s, control_timestep_s=2.0e-4
):
    """Capture controller commands without importing MuJoCo/FlyGym."""

    import fly_sensor2behavior.flybody_adapter as adapter

    def fake_build(config):
        simulation = SimpleNamespace(
            mj_model=object(),
            mj_data=SimpleNamespace(qpos=np.zeros(6), qvel=np.zeros(6)),
        )
        return {
            "config": config,
            "simulation": simulation,
            "wing_qpos_adrs": np.arange(6, dtype=int),
            "wing_dof_adrs": np.arange(6, dtype=int),
        }

    def fake_run(bundle, n_steps, controller, metadata_extra):
        commands = np.vstack(
            [np.asarray(controller(step, bundle), dtype=float) for step in range(n_steps)]
        )
        return SimpleNamespace(commands=commands, metadata=metadata_extra)

    monkeypatch.setattr(adapter, "build_flybody_simulation", fake_build)
    monkeypatch.setattr(adapter, "_run_built_trace", fake_run)
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(mj_forward=lambda _model, _data: None),
    )
    return adapter.run_analytic_wingbeat_smoke(
        duration_s=8.0e-4,
        control_timestep_s=control_timestep_s,
        config=FlyBodyWorkerConfig(timestep_s=physics_timestep_s),
    )


def test_analytic_controller_uses_same_upstream_control_ticks_across_physics_steps(
    monkeypatch,
):
    # Upstream FlyBody uses a 0.2 ms flight-control clock and a 0.05 ms
    # physics clock.  The convergence pair must retain those physical control
    # ticks while refining only the physics integration grid.
    coarse = _capture_analytic_controller(
        monkeypatch, physics_timestep_s=1.0e-4
    )
    fine = _capture_analytic_controller(
        monkeypatch, physics_timestep_s=5.0e-5
    )

    np.testing.assert_allclose(coarse.commands[::2], fine.commands[::4])
    np.testing.assert_allclose(
        coarse.commands.reshape(4, 2, 6),
        np.repeat(coarse.commands[::2, None, :], 2, axis=1),
    )
    np.testing.assert_allclose(
        fine.commands.reshape(4, 4, 6),
        np.repeat(fine.commands[::4, None, :], 4, axis=1),
    )
    assert coarse.metadata["control_timestep_s"] == pytest.approx(2.0e-4)
    assert fine.metadata["control_timestep_s"] == pytest.approx(2.0e-4)
    assert coarse.metadata["actuation_hold"] == "zero_order_hold"
    assert fine.metadata["actuation_hold"] == "zero_order_hold"
    assert coarse.metadata["control_phase_semantics"] == "advance_then_hold"

    from fly_sensor2behavior.flybody_adapter import _analytic_wing_pattern

    initial, _ = _analytic_wing_pattern(0.0, 1.0, 1.0)
    first_target, _ = _analytic_wing_pattern(
        2.0 * np.pi * 218.0 * 2.0e-4, 1.0, 1.0
    )
    np.testing.assert_allclose(
        coarse.commands[0], 1.8e-6 * (first_target - initial)
    )


@pytest.mark.parametrize("control_timestep_s", (7.5e-5, 2.5e-5))
def test_analytic_controller_rejects_nonintegral_or_subphysics_control_clock(
    control_timestep_s,
):
    from fly_sensor2behavior.flybody_adapter import run_analytic_wingbeat_smoke

    # This validation happens before native worker construction, so an invalid
    # multirate schedule cannot be silently rounded by either runtime.
    with pytest.raises(ValueError, match="integer multiple of physics timestep"):
        run_analytic_wingbeat_smoke(
            duration_s=1.0e-3,
            control_timestep_s=control_timestep_s,
            config=FlyBodyWorkerConfig(timestep_s=5.0e-5),
        )


@pytest.mark.skipif(
    dependency_versions()["flygym"] != PINNED_FLYGYM_VERSION
    and os.environ.get("FLY_S2B_REQUIRE_COMPILED_WORKER") != "1",
    reason="pinned FlyBody worker dependencies are not installed",
)
def test_compiled_worker_applies_reviewed_flight_units_and_structure():
    from fly_sensor2behavior.flybody_adapter import build_flybody_simulation

    bundle = build_flybody_simulation()
    model = bundle["simulation"].mj_model
    assert model.nu == 6
    assert model.ntendon == 8
    assert tuple(dof.name for dof in bundle["wing_dofs"]) == (
        REVIEWED_FLYBODY_WING_AXIS_ORDER
    )
    assert bundle["ground_geom_name"] == "ground_plane"
    assert bundle["ground_contact_pair_count"] == 48
    assert model.opt.density == pytest.approx(1.28e-6)
    assert model.opt.viscosity == pytest.approx(1.85e-5)
    assert tuple(model.opt.gravity) == pytest.approx((0.0, 0.0, -9810.0))
    for geom_name in bundle["fluid_geom_names"]:
        geom_id = model.geom(geom_name).id
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0
    for dof in bundle["wing_dofs"]:
        joint = model.joint("flybody/" + dof.name)
        dof_index = int(model.jnt_dofadr[joint.id])
        assert model.jnt_stiffness[joint.id] == pytest.approx(1.0)
        assert model.dof_damping[dof_index] == pytest.approx(0.776923)
    adapter = FlyBodyPhysicsAdapter.__new__(FlyBodyPhysicsAdapter)
    adapter.bundle = bundle
    receipt = adapter.provenance_metadata()
    assert receipt["worker_versions"]["flygym"] == PINNED_FLYGYM_VERSION
    assert receipt["compiled_model_fingerprint"]["sha256"].startswith("sha256:")
    assert receipt["dependency_record_sha256"]["mujoco"].startswith("sha256:")
    assert receipt["released_policy_topology_equivalent"] is False
    assert receipt["ground_contact_telemetry"]["configured_pair_count"] == 48
    assert receipt["checkpoint_contract"] == {
        "schema_version": "1.0.0",
        "state_spec": "mjSTATE_INTEGRATION",
        "float_encoding": "float64_le_base64",
        "compatibility_bound_to_compiled_model": True,
        "fresh_adapter_exact_reentry_required": True,
    }
    assert "pre-integration transition" in receipt["ground_contact_telemetry"][
        "kind"
    ]
    assert "not the post-step pose" in receipt["ground_contact_telemetry"][
        "sample_semantics"
    ]
    assert receipt["licenses"]["flybody_model_assets"].startswith(
        "not independently declared"
    )
