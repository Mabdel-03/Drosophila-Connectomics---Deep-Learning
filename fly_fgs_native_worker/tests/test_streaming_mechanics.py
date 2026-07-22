from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace
from typing import Optional

import numpy as np
import pytest

from fly_sensor2behavior.flight.streaming_bridge import (
    RawAppSide,
    StreamingNOD1MotorBridge,
    StreamingMotorEvent,
)
from fly_sensor2behavior.flight.streaming_mechanics import (
    STREAMING_MECHANICS_DT_S,
    STREAMING_MECHANICS_STEPS_PER_BRIDGE,
    MechanicsInterventionMode,
    StreamingMechanicsCheckpoint,
    StreamingMechanicsConfig,
    StreamingMuscleIntervention,
    StreamingMuscleWingStepper,
    WingPhaseSource,
)
from fly_sensor2behavior.schema import AnatomicalSide


_MOTOR_BY_MUSCLE = {
    "iv2": "MN-iv2",
    "i1": "MN-i1",
    "iv1": "MN-iv1",
    "b3": "MN-b3",
}
_PHASE_BY_MUSCLE = {
    "iv2": 0.20 * 2.0 * math.pi,
    "i1": 0.35 * 2.0 * math.pi,
    "iv1": 0.55 * 2.0 * math.pi,
    "b3": 0.75 * 2.0 * math.pi,
}


def _event(
    event_id: str,
    availability_time_s: float,
    *,
    side: RawAppSide = RawAppSide.L,
    muscle: str = "iv2",
    event_time_s: Optional[float] = None,
) -> StreamingMotorEvent:
    event_time = (
        max(0.0, availability_time_s - 0.0005)
        if event_time_s is None
        else event_time_s
    )
    return StreamingMotorEvent(
        event_id=event_id,
        motor_neuron=_MOTOR_BY_MUSCLE[muscle],
        muscle=muscle,
        raw_app_side=side,
        anatomical_side=AnatomicalSide.UNKNOWN,
        event_time_s=event_time,
        availability_time_s=availability_time_s,
        wingbeat_phase_rad=_PHASE_BY_MUSCLE[muscle],
        rate_hz=200.0,
        emission_probability=1.0,
        source_measurement_time_s=0.0,
        generator_seed=17,
        phase_crossing_index=0,
    )


def _canonical_digest(payload) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _intervention(
    intervention_id: str,
    *,
    muscle: str = "iv2",
    start_s: float = 0.0001,
    end_s: float = 0.0003,
    mode: MechanicsInterventionMode = MechanicsInterventionMode.SILENCE,
    side: Optional[RawAppSide] = RawAppSide.L,
    output_scale: float = 0.0,
) -> StreamingMuscleIntervention:
    return StreamingMuscleIntervention(
        intervention_id=intervention_id,
        muscle=muscle,
        start_s=start_s,
        end_s=end_s,
        mode=mode,
        raw_app_side=side,
        output_scale=output_scale,
    )


def _assert_wings_exact(first, second) -> None:
    assert first.phase_rad == second.phase_rad
    assert first.frequency_hz == second.frequency_hz
    for name in (
        "stroke_rad",
        "stroke_velocity_rad_s",
        "stroke_acceleration_rad_s2",
        "angle_of_attack_rad",
        "deviation_rad",
        "generalized_torque_n_m",
        "wing_axis_torque_n_m",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def _advance_five_unbound(stepper, events=()):
    """Component-test helper; canonical composition uses a bridge receipt."""

    stepper.push_delivered_events(events)
    return tuple(stepper.step() for _ in range(STREAMING_MECHANICS_STEPS_PER_BRIDGE))


def test_explicit_uncalibrated_carrier_and_no_local_event_generator() -> None:
    stepper = StreamingMuscleWingStepper()
    frame = stepper.step()

    assert STREAMING_MECHANICS_DT_S == 0.0001
    assert STREAMING_MECHANICS_STEPS_PER_BRIDGE == 5
    assert frame.phase_source is WingPhaseSource.MODEL_OWNED_OSCILLATOR
    assert "uncalibrated" in frame.carrier_baseline_status
    assert "not biological stable flight" in frame.carrier_baseline_status
    assert "anatomical laterality remains unknown" in (
        frame.raw_lane_to_virtual_wing_convention
    )
    assert set(frame.muscle_snapshot.individual) == {
        "%s:%s" % (side, muscle)
        for side in ("left", "right")
        for muscle in ("DLM", "DVM", "tp1", "iv2", "i1", "iv1", "b3")
    }
    assert frame.muscle_snapshot.power_left == frame.muscle_snapshot.power_right
    assert frame.muscle_snapshot.tension_left == frame.muscle_snapshot.tension_right
    assert "seed" not in inspect.signature(StreamingMuscleWingStepper).parameters
    assert stepper.applied_event_ids == ()


def test_nmj_availability_and_right_boundary_are_strictly_causal() -> None:
    baseline = StreamingMuscleWingStepper()
    expected_first = baseline.step()
    expected_second = baseline.step()

    stepper = StreamingMuscleWingStepper()
    # The event was generated at t=0 but is unavailable until the exact right
    # endpoint of [0, 0.1 ms), so the first interval cannot observe it.
    event = _event("boundary", 0.0001, event_time_s=0.0)
    stepper.push_delivered_events((event,))
    first = stepper.step()
    assert first.applied_event_ids == ()
    assert first.pending_event_count == 1
    _assert_wings_exact(first.wing_kinematics, expected_first.wing_kinematics)

    second = stepper.step()
    assert second.applied_event_ids == ("boundary",)
    assert second.pending_event_count == 0
    assert second.muscle_snapshot.individual["left:iv2"].activation > 0.0
    assert second.muscle_snapshot.individual["right:iv2"].activation == 0.0
    assert not np.array_equal(
        second.wing_kinematics.angle_of_attack_rad,
        expected_second.wing_kinematics.angle_of_attack_rad,
    )

    for _ in range(4):
        assert "boundary" not in stepper.step().applied_event_ids
    assert stepper.applied_event_ids == ("boundary",)

    # Integer-grid reconstruction can spell 0.3 ms as adjacent binary floats;
    # the endpoint convention must remain stable beyond the first tick.
    later = StreamingMuscleWingStepper()
    later.push_delivered_events((_event("later-boundary", 0.0003),))
    assert later.step().applied_event_ids == ()
    assert later.step().applied_event_ids == ()
    assert later.step().applied_event_ids == ()
    assert later.step().applied_event_ids == ("later-boundary",)


def test_subgrid_nmj_availability_changes_state_only_after_its_exact_time() -> None:
    baseline = StreamingMuscleWingStepper()
    baseline_first = baseline.step()
    baseline_second = baseline.step()
    early = StreamingMuscleWingStepper()
    late = StreamingMuscleWingStepper()
    early.push_delivered_events((_event("early", 0.0),))
    late.push_delivered_events((_event("late", 0.00005),))

    early_frame = early.step()
    late_frame = late.step()
    assert early_frame.applied_event_ids == ("early",)
    assert late_frame.applied_event_ids == ("late",)
    # The late event has had only 0.05 ms to decay by the endpoint, while the
    # event at the left edge has had the full 0.1 ms.
    assert (
        late_frame.muscle_snapshot.individual["left:iv2"].activation
        > early_frame.muscle_snapshot.individual["left:iv2"].activation
    )
    # Neither an event at the left boundary nor one inside the interval can
    # alter a command already defined for that same physics interval.
    _assert_wings_exact(
        early_frame.actuation_wing_kinematics,
        baseline_first.actuation_wing_kinematics,
    )
    _assert_wings_exact(
        late_frame.actuation_wing_kinematics,
        baseline_first.actuation_wing_kinematics,
    )
    early_second = early.step()
    assert not np.array_equal(
        early_second.actuation_wing_kinematics.angle_of_attack_rad,
        baseline_second.actuation_wing_kinematics.angle_of_attack_rad,
    )
    assert "left-boundary zero-order-hold" in (
        early_frame.physics_command_interval_semantics
    )


def test_duplicate_or_retroactive_events_are_rejected_transactionally() -> None:
    stepper = StreamingMuscleWingStepper()
    duplicate_a = _event("same", 0.0)
    duplicate_b = _event("same", 0.00005, muscle="i1")
    with pytest.raises(ValueError, match="duplicate delivered event ID"):
        stepper.push_delivered_events((duplicate_a, duplicate_b))
    assert stepper.pending_event_count == 0

    stepper.push_delivered_events((duplicate_a,))
    before = stepper.checkpoint().to_dict()
    with pytest.raises(ValueError, match="duplicate delivered event ID"):
        stepper.push_delivered_events((duplicate_a,))
    assert stepper.checkpoint().to_dict() == before
    stepper.step()
    with pytest.raises(ValueError, match="duplicate delivered event ID"):
        stepper.push_delivered_events((duplicate_a,))

    stepper.step()
    before = stepper.checkpoint().to_dict()
    with pytest.raises(ValueError, match="retroactively"):
        stepper.push_delivered_events((_event("old", 0.0),))
    assert stepper.checkpoint().to_dict() == before


def test_raw_app_lane_swap_is_mirror_symmetric_without_anatomy_claim() -> None:
    raw_l = StreamingMuscleWingStepper()
    raw_r = StreamingMuscleWingStepper()
    left_event = _event("raw-l", 0.0, side=RawAppSide.L, muscle="i1")
    right_event = _event("raw-r", 0.0, side=RawAppSide.R, muscle="i1")
    assert left_event.anatomical_side is AnatomicalSide.UNKNOWN
    assert right_event.anatomical_side is AnatomicalSide.UNKNOWN
    raw_l.push_delivered_events((left_event,))
    raw_r.push_delivered_events((right_event,))

    left_wings = raw_l.step().wing_kinematics
    right_wings = raw_r.step().wing_kinematics
    for name in (
        "stroke_rad",
        "stroke_velocity_rad_s",
        "stroke_acceleration_rad_s2",
        "angle_of_attack_rad",
        "deviation_rad",
        "generalized_torque_n_m",
    ):
        np.testing.assert_array_equal(
            getattr(left_wings, name), getattr(right_wings, name)[::-1]
        )
    np.testing.assert_array_equal(
        left_wings.wing_axis_torque_n_m[:3],
        right_wings.wing_axis_torque_n_m[3:],
    )
    np.testing.assert_array_equal(
        left_wings.wing_axis_torque_n_m[3:],
        right_wings.wing_axis_torque_n_m[:3],
    )


def test_model_phase_projection_is_nonmutating_exact_and_event_independent() -> None:
    stepper = StreamingMuscleWingStepper()
    stepper.push_delivered_events(
        (
            _event("iv2", 0.00002, muscle="iv2"),
            _event("b3", 0.00043, side=RawAppSide.R, muscle="b3"),
        )
    )
    before = stepper.checkpoint().to_dict()
    projected = stepper.project_phase_end_unwrapped_rad()
    assert stepper.checkpoint().to_dict() == before

    frames = _advance_five_unbound(stepper)
    assert len(frames) == 5
    assert stepper.phase_unwrapped_rad == projected
    assert frames[-1].phase_end_unwrapped_rad == projected
    assert stepper.applied_event_ids == ("iv2", "b3")

    empty = StreamingMuscleWingStepper()
    assert empty.project_phase_end_unwrapped_rad() == projected
    _advance_five_unbound(empty)
    assert empty.phase_unwrapped_rad == projected

    with pytest.raises(ValueError, match="0.1 ms grid"):
        stepper.project_phase_end_unwrapped_rad(0.00055)


def test_measured_unwrapped_phase_is_explicit_and_model_phase_rejects_it() -> None:
    model_owned = StreamingMuscleWingStepper()
    before = model_owned.checkpoint().to_dict()
    with pytest.raises(ValueError, match="forbidden"):
        model_owned.step(measured_phase_end_unwrapped_rad=0.1)
    assert model_owned.checkpoint().to_dict() == before

    measured = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(phase_source=WingPhaseSource.MEASURED_UNWRAPPED)
    )
    with pytest.raises(ValueError, match="requires every endpoint"):
        measured.step()
    endpoint = 2.0 * math.pi * 200.0 * STREAMING_MECHANICS_DT_S
    frame = measured.step(measured_phase_end_unwrapped_rad=endpoint)
    assert frame.phase_source is WingPhaseSource.MEASURED_UNWRAPPED
    assert frame.phase_end_unwrapped_rad == endpoint
    assert frame.wing_kinematics.phase_rad == endpoint
    assert frame.kinematic_frequency_hz == pytest.approx(200.0)
    assert frame.model_owned_frequency_hz > 0.0
    with pytest.raises(ValueError, match="only for model-owned"):
        measured.project_phase_end_unwrapped_rad()


def test_checkpoint_fresh_instance_replays_exact_tail_and_pending_events_once() -> None:
    uninterrupted = StreamingMuscleWingStepper()
    uninterrupted.push_delivered_events(
        (
            _event("a", 0.00015, muscle="iv2"),
            _event("b", 0.00040, side=RawAppSide.R, muscle="i1"),
            _event("c", 0.00060, muscle="b3"),
        )
    )
    for _ in range(3):
        uninterrupted.step()
    checkpoint = uninterrupted.checkpoint()
    assert checkpoint.to_dict()["state"]["pending_events"]

    expected = [uninterrupted.step() for _ in range(6)]
    resumed = StreamingMuscleWingStepper.from_checkpoint(checkpoint)
    actual = [resumed.step() for _ in range(6)]
    for expected_frame, actual_frame in zip(expected, actual):
        assert actual_frame.tick_index == expected_frame.tick_index
        assert actual_frame.interval_start_s == expected_frame.interval_start_s
        assert actual_frame.interval_end_s == expected_frame.interval_end_s
        assert actual_frame.phase_start_unwrapped_rad == expected_frame.phase_start_unwrapped_rad
        assert actual_frame.phase_end_unwrapped_rad == expected_frame.phase_end_unwrapped_rad
        assert actual_frame.applied_event_ids == expected_frame.applied_event_ids
        _assert_wings_exact(
            actual_frame.wing_kinematics, expected_frame.wing_kinematics
        )
    assert resumed.checkpoint().to_dict() == uninterrupted.checkpoint().to_dict()
    assert resumed.applied_event_ids == ("a", "b", "c")


def test_corrupt_cross_config_and_semantic_checkpoints_fail_before_mutation() -> None:
    source = StreamingMuscleWingStepper()
    source.push_delivered_events((_event("future", 0.0004),))
    source.step()
    checkpoint = source.checkpoint()

    corrupt = checkpoint.to_dict()
    corrupt["state"]["tick_index"] = 99
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        StreamingMechanicsCheckpoint(corrupt)

    incompatible = StreamingMuscleWingStepper(
        replace(
            StreamingMechanicsConfig(),
            baseline_power_activation_bias=0.19,
        )
    )
    before = incompatible.checkpoint().to_dict()
    with pytest.raises(ValueError, match="config mismatch"):
        incompatible.restore(checkpoint)
    assert incompatible.checkpoint().to_dict() == before

    semantic = checkpoint.to_dict()
    semantic["state"]["power"][0]["state"]["activation"] = 99.0
    semantic["payload_sha256"] = _canonical_digest(semantic)
    rehashed = StreamingMechanicsCheckpoint(semantic)
    target = StreamingMuscleWingStepper()
    before = target.checkpoint().to_dict()
    with pytest.raises(ValueError, match="power activation"):
        target.restore(rehashed)
    assert target.checkpoint().to_dict() == before

    forged_carrier = checkpoint.to_dict()
    forged_carrier["state"]["phase_unwrapped_rad"] += 0.01
    forged_carrier["state"]["oscillator"]["phase_rad"] += 0.01
    forged_carrier["payload_sha256"] = _canonical_digest(forged_carrier)
    with pytest.raises(ValueError, match="carrier state is inconsistent"):
        target.restore(StreamingMechanicsCheckpoint(forged_carrier))
    assert target.checkpoint().to_dict() == before

    applied_source = StreamingMuscleWingStepper()
    applied_source.push_delivered_events((_event("applied", 0.0),))
    applied_source.step()
    forged_ledger = applied_source.checkpoint().to_dict()
    forged_ledger["state"]["seen_event_ids"].append("invented")
    forged_ledger["state"]["applied_event_ids"].append("invented")
    forged_ledger["payload_sha256"] = _canonical_digest(forged_ledger)
    with pytest.raises(ValueError, match="receipts do not match"):
        target.restore(StreamingMechanicsCheckpoint(forged_ledger))
    assert target.checkpoint().to_dict() == before


def test_no_event_loss_across_successive_five_to_one_bridge_intervals() -> None:
    stepper = StreamingMuscleWingStepper()
    first_batch = tuple(
        _event(
            "first-%d" % index,
            index * STREAMING_MECHANICS_DT_S,
            side=RawAppSide.L if index % 2 == 0 else RawAppSide.R,
            muscle=tuple(_MOTOR_BY_MUSCLE)[index % 4],
        )
        for index in range(5)
    )
    first_frames = _advance_five_unbound(stepper, first_batch)
    assert [frame.interval_end_s for frame in first_frames] == pytest.approx(
        [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    )
    assert sum(len(frame.applied_event_ids) for frame in first_frames) == 5
    assert first_frames[-1].pending_event_count == 0

    second_batch = tuple(
        _event(
            "second-%d" % index,
            (5 + index) * STREAMING_MECHANICS_DT_S,
            side=RawAppSide.R if index % 2 == 0 else RawAppSide.L,
            muscle=tuple(_MOTOR_BY_MUSCLE)[index % 4],
        )
        for index in range(5)
    )
    second_frames = _advance_five_unbound(stepper, second_batch)
    assert second_frames[-1].interval_end_s == pytest.approx(0.001)
    assert sum(len(frame.applied_event_ids) for frame in second_frames) == 5
    assert second_frames[-1].pending_event_count == 0
    assert set(stepper.applied_event_ids) == {
        *(event.event_id for event in first_batch),
        *(event.event_id for event in second_batch),
    }
    assert len(stepper.applied_event_ids) == 10


def test_authoritative_bridge_receipt_enforces_exact_five_to_one_clock() -> None:
    bridge = StreamingNOD1MotorBridge()
    stepper = StreamingMuscleWingStepper()

    interval_zero = bridge.begin_interval()
    frames_zero = stepper.advance_bridge_interval(interval_zero)
    assert len(frames_zero) == STREAMING_MECHANICS_STEPS_PER_BRIDGE
    phase_path_zero = (
        interval_zero.wing_phase_start_unwrapped_rad,
        *(frame.phase_end_unwrapped_rad for frame in frames_zero),
    )
    bridge.end_interval(phase_path_zero)

    with pytest.raises(ValueError, match="does not match mechanics clock"):
        stepper.advance_bridge_interval(interval_zero)
    with pytest.raises(TypeError, match="StreamingBridgeIntervalStart"):
        stepper.advance_bridge_interval(())

    interval_one = bridge.begin_interval()
    frames_one = stepper.advance_bridge_interval(interval_one)
    assert frames_one[0].tick_index == STREAMING_MECHANICS_STEPS_PER_BRIDGE
    bridge.end_interval(
        (
            interval_one.wing_phase_start_unwrapped_rad,
            *(frame.phase_end_unwrapped_rad for frame in frames_one),
        )
    )


def test_intervention_contract_is_grid_aligned_hashed_and_config_bound() -> None:
    intervention = _intervention("left-iv2-silence")
    assert intervention.active(0.0001)
    assert not intervention.active(0.0003)
    assert len(intervention.contract_sha256) == 64
    assert "anatomical" not in intervention.to_dict()

    config = StreamingMechanicsConfig(muscle_interventions=(intervention,))
    restored = StreamingMechanicsConfig.from_dict(config.to_dict())
    assert restored == config
    assert (
        StreamingMuscleWingStepper(config).intervention_schedule_sha256
        == StreamingMuscleWingStepper(restored).intervention_schedule_sha256
    )
    with pytest.raises(ValueError, match="exact 0.1 ms grid"):
        _intervention("off-grid", start_s=0.00015)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _intervention(
            "amplification-is-not-this-contract",
            mode=MechanicsInterventionMode.SCALE,
            output_scale=1.1,
        )


def test_silence_suppresses_pending_nmj_event_exactly_at_onset() -> None:
    silence = _intervention("silence-onset", start_s=0.0001, end_s=0.0003)
    stepper = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(muscle_interventions=(silence,))
    )
    stepper.push_delivered_events((_event("pending-at-onset", 0.0001),))

    before = stepper.step()
    assert before.applied_event_ids == ()
    assert before.suppressed_event_ids == ()
    assert before.pending_event_count == 1

    onset = stepper.step()
    assert onset.active_intervention_ids == ("silence-onset",)
    assert onset.applied_event_ids == ()
    assert onset.suppressed_event_ids == ("pending-at-onset",)
    assert stepper.applied_event_ids == ()
    assert stepper.suppressed_event_ids == ("pending-at-onset",)
    assert onset.natural_muscle_snapshot.individual["left:iv2"].activation == 0.0
    assert onset.actuation_muscle_snapshot.individual["left:iv2"].force_n == 0.0
    assert onset.pending_event_count == 0


def test_silence_immediately_gates_preexisting_force_but_natural_state_decays() -> None:
    silence = _intervention("force-cut", start_s=0.0001, end_s=0.0004)
    stepper = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(muscle_interventions=(silence,))
    )
    stepper.push_delivered_events(
        (
            _event("preexisting-target", 0.0, muscle="iv2"),
            _event("preexisting-control", 0.0, muscle="i1"),
        )
    )
    buildup = stepper.step()
    target_before = buildup.natural_muscle_snapshot.individual["left:iv2"]
    assert target_before.force_n > 0.0

    gated = stepper.step()
    effective_target = gated.actuation_muscle_snapshot.individual["left:iv2"]
    effective_control = gated.actuation_muscle_snapshot.individual["left:i1"]
    natural_target = gated.natural_muscle_snapshot.individual["left:iv2"]
    assert gated.active_intervention_ids == ("force-cut",)
    assert effective_target.activation == 0.0
    assert effective_target.force_n == 0.0
    assert effective_control.force_n > 0.0
    assert 0.0 < natural_target.activation < target_before.activation
    assert 0.0 < natural_target.force_n < target_before.force_n
    # The target's virtual hinge-axis contribution is gone at onset, while the
    # non-targeted i1 contribution remains in the same raw coordinate lane.
    assert gated.actuation_wing_kinematics.wing_axis_torque_n_m[0] != 0.0
    assert np.count_nonzero(gated.actuation_wing_kinematics.wing_axis_torque_n_m[:3]) > 0


def test_half_open_end_boundary_restores_output_and_accepts_nmj_excitation() -> None:
    silence = _intervention("one-tick-cut", start_s=0.0001, end_s=0.0002)
    stepper = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(muscle_interventions=(silence,))
    )
    stepper.push_delivered_events(
        (
            _event("preload", 0.0),
            _event("at-start", 0.0001),
            _event("at-end", 0.0002),
        )
    )
    stepper.step()
    active = stepper.step()
    restored = stepper.step()

    assert active.active_intervention_ids == ("one-tick-cut",)
    assert active.suppressed_event_ids == ("at-start",)
    # The endpoint snapshot is the command basis for the next interval, so the
    # half-open end boundary is already ungated there.
    assert active.actuation_muscle_snapshot.individual["left:iv2"].force_n == 0.0
    assert active.muscle_snapshot.individual["left:iv2"].force_n > 0.0
    assert restored.active_intervention_ids == ()
    assert restored.applied_event_ids == ("at-end",)
    assert restored.suppressed_event_ids == ()
    assert restored.natural_muscle_snapshot.individual["left:iv2"].force_n > 0.0


def test_scale_only_gates_target_output_without_changing_hidden_state() -> None:
    scale = _intervention(
        "quarter-iv2",
        start_s=0.0,
        end_s=0.0003,
        mode=MechanicsInterventionMode.SCALE,
        output_scale=0.25,
    )
    gated = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(muscle_interventions=(scale,))
    )
    control = StreamingMuscleWingStepper()
    events = (
        _event("target", 0.0, muscle="iv2"),
        _event("same-lane-control", 0.0, muscle="i1"),
        _event("other-lane-control", 0.0, side=RawAppSide.R, muscle="iv2"),
    )
    gated.push_delivered_events(events)
    control.push_delivered_events(events)
    gated_frame = gated.step()
    control_frame = control.step()

    assert gated_frame.suppressed_event_ids == ()
    assert gated_frame.applied_event_ids == control_frame.applied_event_ids
    assert gated_frame.natural_muscle_snapshot == control_frame.natural_muscle_snapshot
    target_natural = gated_frame.natural_muscle_snapshot.individual["left:iv2"]
    target_effective = gated_frame.muscle_snapshot.individual["left:iv2"]
    assert target_effective.activation == pytest.approx(0.25 * target_natural.activation)
    assert target_effective.force_n == pytest.approx(0.25 * target_natural.force_n)
    assert (
        gated_frame.muscle_snapshot.individual["left:i1"]
        == gated_frame.natural_muscle_snapshot.individual["left:i1"]
    )
    assert (
        gated_frame.muscle_snapshot.individual["right:iv2"]
        == gated_frame.natural_muscle_snapshot.individual["right:iv2"]
    )


def test_intervention_checkpoint_fresh_instance_continuation_is_exact() -> None:
    silence = _intervention("checkpoint-cut", start_s=0.0001, end_s=0.0006)
    source = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(muscle_interventions=(silence,))
    )
    source.push_delivered_events(
        (
            _event("suppressed", 0.0001, muscle="iv2"),
            _event("nontarget", 0.0002, muscle="i1"),
            _event("after-cut", 0.0006, muscle="iv2"),
        )
    )
    for _ in range(3):
        source.step()
    checkpoint = source.checkpoint()
    checkpoint_state = checkpoint.to_dict()["state"]
    assert checkpoint_state["active_intervention_ids"] == ["checkpoint-cut"]
    assert checkpoint_state["suppressed_event_ids"] == ["suppressed"]

    expected = [source.step() for _ in range(5)]
    resumed = StreamingMuscleWingStepper.from_checkpoint(checkpoint)
    actual = [resumed.step() for _ in range(5)]
    for expected_frame, actual_frame in zip(expected, actual):
        assert actual_frame.tick_index == expected_frame.tick_index
        assert actual_frame.applied_event_ids == expected_frame.applied_event_ids
        assert actual_frame.suppressed_event_ids == expected_frame.suppressed_event_ids
        assert (
            actual_frame.active_intervention_ids
            == expected_frame.active_intervention_ids
        )
        assert (
            actual_frame.actuation_muscle_snapshot
            == expected_frame.actuation_muscle_snapshot
        )
        assert (
            actual_frame.natural_muscle_snapshot
            == expected_frame.natural_muscle_snapshot
        )
        _assert_wings_exact(
            actual_frame.actuation_wing_kinematics,
            expected_frame.actuation_wing_kinematics,
        )
        _assert_wings_exact(
            actual_frame.wing_kinematics, expected_frame.wing_kinematics
        )
    assert resumed.checkpoint().to_dict() == source.checkpoint().to_dict()
    assert resumed.suppressed_event_ids == ("suppressed",)
    assert resumed.applied_event_ids == ("nontarget", "after-cut")


def test_rehashed_intervention_checkpoint_corruption_is_semantically_rejected() -> None:
    silence = _intervention("semantic-cut", start_s=0.0001, end_s=0.0004)
    config = StreamingMechanicsConfig(muscle_interventions=(silence,))
    source = StreamingMuscleWingStepper(config)
    source.push_delivered_events((_event("was-suppressed", 0.0001),))
    source.step()
    source.step()
    checkpoint = source.checkpoint()
    target = StreamingMuscleWingStepper(config)
    before = target.checkpoint().to_dict()

    bad_active = checkpoint.to_dict()
    bad_active["state"]["active_intervention_ids"] = []
    bad_active["payload_sha256"] = _canonical_digest(bad_active)
    with pytest.raises(ValueError, match="active intervention IDs are inconsistent"):
        target.restore(StreamingMechanicsCheckpoint(bad_active))
    assert target.checkpoint().to_dict() == before

    relabelled = checkpoint.to_dict()
    relabelled["state"]["applied_event_ids"] = relabelled["state"].pop(
        "suppressed_event_ids"
    )
    relabelled["state"]["applied_events"] = relabelled["state"].pop(
        "suppressed_events"
    )
    relabelled["state"]["suppressed_event_ids"] = []
    relabelled["state"]["suppressed_events"] = []
    relabelled["payload_sha256"] = _canonical_digest(relabelled)
    with pytest.raises(ValueError, match="during an active SILENCE"):
        target.restore(StreamingMechanicsCheckpoint(relabelled))
    assert target.checkpoint().to_dict() == before

    bad_contract = checkpoint.to_dict()
    bad_contract["config"]["muscle_interventions"][0]["end_s"] = 0.0005
    bad_contract["payload_sha256"] = _canonical_digest(bad_contract)
    with pytest.raises(ValueError, match="intervention contract SHA-256 mismatch"):
        target.restore(StreamingMechanicsCheckpoint(bad_contract))
    assert target.checkpoint().to_dict() == before
