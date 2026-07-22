from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest

from fly_sensor2behavior.fly_fgs import (
    FLY_FGS_NOD1_RAW_APP_SIDES,
    FLY_FGS_NOD1_ROOT_IDS,
)
from fly_sensor2behavior.fly_fgs_runtime import (
    FlyFGSCircuitSample,
    FlyFGSSceneBodyInput,
)
from fly_sensor2behavior.flight.bridge import (
    CircuitToDNEncoder,
    CircuitToDNEncoderConfig,
    VNCMotorNetwork,
    VNCMotorNetworkConfig,
)
from fly_sensor2behavior.flight.streaming_bridge import (
    RawAppSide,
    STREAMING_BRIDGE_DT_S,
    StreamingBridgeCheckpoint,
    StreamingBridgeConfig,
    StreamingBridgeIntervalStart,
    StreamingBridgeIntervention,
    StreamingBridgeStage,
    StreamingInterventionMode,
    StreamingNOD1MotorBridge,
    default_streaming_motor_pathways,
)
from fly_sensor2behavior.schema import (
    AnatomicalSide,
    CircuitOutputTrace,
    CircuitSignal,
    CircuitSignalKind,
    Confidence,
    ConfidenceLevel,
    DatasetNamespace,
    DatasetRef,
    EntityKind,
    EntityRef,
    EvidenceTier,
    EyeSide,
    Provenance,
    SideContext,
    SideMappingMethod,
)


_TWO_PI = 2.0 * math.pi


def _control(offset: float = 0.0) -> FlyFGSSceneBodyInput:
    return FlyFGSSceneBodyInput(
        heading_rad=offset,
        heading_velocity_rad_s=offset,
        figure_world_azimuth_rad=offset,
        figure_velocity_rad_s=offset,
        ground_velocity_rad_s=offset,
    )


def _sample(
    sample_index: int,
    measurement_time_s: float,
    availability_time_s: float,
    *,
    raw_l_voltage_v: float = -0.055,
    raw_r_voltage_v: float = -0.060,
    pooled_readout=None,
    full_cell_voltage_v=None,
    full_cell_activity=None,
    retinal_input_luminance=None,
    control_offset: float = 0.0,
) -> FlyFGSCircuitSample:
    voltages = {
        root_id: (
            raw_l_voltage_v
            if FLY_FGS_NOD1_RAW_APP_SIDES[root_id] == "L"
            else raw_r_voltage_v
        )
        for root_id in FLY_FGS_NOD1_ROOT_IDS
    }
    return FlyFGSCircuitSample(
        sample_index=sample_index,
        measurement_time_s=measurement_time_s,
        availability_time_s=availability_time_s,
        nod1_voltage_v=voltages,
        pooled_readout={} if pooled_readout is None else pooled_readout,
        last_control=_control(control_offset),
        retinal_input_luminance=retinal_input_luminance,
        full_cell_voltage_v=full_cell_voltage_v,
        full_cell_activity=full_cell_activity,
        full_cell_state_eligible_motor_input=False,
    )


def _rate_by_side(samples):
    return {sample.raw_app_side: sample.rate_hz for sample in samples}


def _motor_rate(samples, side: RawAppSide, target: str) -> float:
    matches = [
        sample
        for sample in samples
        if sample.raw_app_side is side and sample.target_name == target
    ]
    assert len(matches) == 1
    return matches[0].rate_hz


def _unit_gain_config(**overrides) -> StreamingBridgeConfig:
    values = {
        "encoder_delay_s": 0.0,
        "vnc_delay_s": 0.0,
        "nmj_delay_s": 0.0005,
        "dn_functional_gain_hz": 200.0,
        "pathways": tuple(
            replace(path, functional_rate_gain=1.0)
            for path in default_streaming_motor_pathways()
        ),
    }
    values.update(overrides)
    return StreamingBridgeConfig(**values)


def test_causal_delays_zero_order_holds_and_raw_app_side_semantics() -> None:
    bridge = StreamingNOD1MotorBridge(seed=9)
    bridge.push_circuit_sample(
        _sample(
            0,
            0.0,
            0.0,
            raw_l_voltage_v=-0.055,
            raw_r_voltage_v=-0.060,
        )
    )
    # Canonical circuit samples arrive on the exact 5 ms fly-FGS timebase.
    bridge.push_circuit_sample(
        _sample(
            1,
            0.005,
            0.005,
            raw_l_voltage_v=-0.060,
            raw_r_voltage_v=-0.060,
        )
    )

    frames = [bridge.step(0.0) for _ in range(15)]
    assert STREAMING_BRIDGE_DT_S == 0.0005
    assert len(frames[0].source_hold) == 4
    assert all(
        hold.anatomical_side is AnatomicalSide.UNKNOWN
        for hold in frames[0].source_hold
    )

    # A raw-app L NOD1 drive enters the contralateral raw-app R DN lane. Raw
    # labels are never promoted to anatomy.
    assert _rate_by_side(frames[0].generated_dn_rates) == pytest.approx(
        {RawAppSide.L: 0.0, RawAppSide.R: 40.0}
    )
    assert all(
        sample.anatomical_side is AnatomicalSide.UNKNOWN
        for frame in frames
        for sample in frame.generated_dn_rates + frame.generated_motor_rates
    )

    # Encoder delay: the 0 ms source update is not held by DNp26 until 3 ms.
    assert RawAppSide.R not in _rate_by_side(frames[5].held_dn_rates)
    assert _rate_by_side(frames[6].held_dn_rates)[RawAppSide.R] == pytest.approx(
        40.0
    )
    # VNC delay: that DN update cannot affect MN-iv2 until 5 ms.
    assert _motor_rate(frames[9].held_motor_rates, RawAppSide.R, "MN-iv2") == 0.0
    assert _motor_rate(
        frames[10].held_motor_rates, RawAppSide.R, "MN-iv2"
    ) == pytest.approx(24.0)

    # Zero-order hold and no future read: source sample 0 remains active through
    # 4.5 ms, then sample 1 changes the drive exactly at 5 ms.
    assert _rate_by_side(frames[9].generated_dn_rates)[RawAppSide.R] == pytest.approx(
        40.0
    )
    assert _rate_by_side(frames[10].generated_dn_rates)[RawAppSide.R] == 0.0


def test_only_four_nod1_voltages_cross_the_upstream_boundary() -> None:
    nod1_a = _sample(
        0,
        0.0,
        0.0,
        raw_l_voltage_v=-0.054,
        raw_r_voltage_v=-0.058,
        pooled_readout={"forbidden_downstream": 1.0},
        retinal_input_luminance=(0.0, 1.0),
        full_cell_voltage_v=(-0.1, -0.2),
        full_cell_activity=(1.0, 2.0),
        control_offset=1.0,
    )
    nod1_b = _sample(
        0,
        0.0,
        0.0,
        raw_l_voltage_v=-0.054,
        raw_r_voltage_v=-0.058,
        pooled_readout={"forbidden_downstream": 999999.0, "toy_yaw": -42.0},
        retinal_input_luminance=(999.0,),
        full_cell_voltage_v=(10.0, 20.0, 30.0),
        full_cell_activity=(99.0,),
        control_offset=-3.0,
    )
    first = StreamingNOD1MotorBridge(seed=123)
    second = StreamingNOD1MotorBridge(seed=123)
    frame_a = first.step(0.11 * _TWO_PI, circuit_sample=nod1_a)
    frame_b = second.step(0.11 * _TWO_PI, circuit_sample=nod1_b)

    assert frame_a == frame_b
    assert first.checkpoint().to_dict() == second.checkpoint().to_dict()
    checkpoint_text = json.dumps(first.checkpoint().to_dict(), sort_keys=True)
    assert "forbidden_downstream" not in checkpoint_text
    assert "toy_yaw" not in checkpoint_text


def test_runtime_phase_crossings_gate_all_individual_steering_events() -> None:
    config = _unit_gain_config()
    bridge = StreamingNOD1MotorBridge(config, seed=5)
    bridge.push_circuit_sample(
        _sample(0, 0.0, 0.0, raw_l_voltage_v=-0.055, raw_r_voltage_v=-0.055)
    )

    # A high motor rate alone emits nothing while the supplied phase is still.
    still = bridge.step(0.0)
    assert [sample.rate_hz for sample in still.held_motor_rates] == pytest.approx(
        [200.0] * 8
    )
    assert still.generated_events == ()
    assert bridge.step(0.0).generated_events == ()

    # The actual runtime phase crosses all four preferred phases in this
    # interval. With p=1 each individual muscle emits once on each raw app lane.
    frame = bridge.step(0.90 * _TWO_PI)
    assert len(frame.generated_events) == 8
    assert {
        (event.muscle, event.raw_app_side)
        for event in frame.generated_events
    } == {
        (muscle, side)
        for muscle in ("iv2", "i1", "iv1", "b3")
        for side in (RawAppSide.L, RawAppSide.R)
    }
    path_by_muscle = {path.muscle: path for path in config.pathways}
    for event in frame.generated_events:
        expected_fraction = (
            path_by_muscle[event.muscle].preferred_phase_rad / (0.90 * _TWO_PI)
        )
        assert event.event_time_s == pytest.approx(
            frame.interval_start_s + expected_fraction * config.base_dt_s
        )
        assert event.wingbeat_phase_rad == pytest.approx(
            path_by_muscle[event.muscle].preferred_phase_rad
        )
        assert event.emission_probability == pytest.approx(1.0)
        assert event.anatomical_side is AnatomicalSide.UNKNOWN


def test_checkpoint_restores_pending_nmj_events_exactly_once() -> None:
    config = _unit_gain_config(nmj_delay_s=0.002)
    intervention = StreamingBridgeIntervention(
        intervention_id="active-iv2-control",
        target_stage=StreamingBridgeStage.MN,
        target_name="MN-iv2",
        raw_app_side=None,
        mode=StreamingInterventionMode.SCALE,
        start_s=0.0,
        end_s=0.010,
        magnitude=1.0,
    )
    uninterrupted = StreamingNOD1MotorBridge(
        config, seed=17, interventions=(intervention,)
    )
    sample = _sample(
        0, 0.0, 0.0, raw_l_voltage_v=-0.055, raw_r_voltage_v=-0.055
    )
    first = uninterrupted.step(0.21 * _TWO_PI, circuit_sample=sample)
    assert len(first.generated_events) == 2
    assert first.delivered_events == ()
    assert first.pending_event_count == 2

    checkpoint = uninterrupted.checkpoint()
    state = checkpoint.to_dict()["state"]
    assert len(state["event_pending"]) == 2
    assert state["delivered_events"] == []
    assert state["active_intervention_ids"] == ["active-iv2-control"]

    # Continue through additional preferred-phase crossings so equality also
    # proves that per-channel RNG state resumes exactly, not merely the queues.
    phase_schedule = [
        (0.21 + 0.85 * step_index) * _TWO_PI
        for step_index in range(1, 7)
    ]
    expected_frames = [uninterrupted.step(phase) for phase in phase_schedule]

    resumed = StreamingNOD1MotorBridge(
        config,
        seed=17,
        interventions=(intervention,),
        initial_wing_phase_unwrapped_rad=123.0,
    )
    resumed.restore(checkpoint)
    actual_frames = [resumed.step(phase) for phase in phase_schedule]
    assert actual_frames == expected_frames
    assert resumed.checkpoint().to_dict() == uninterrupted.checkpoint().to_dict()

    event_ids = [event.event_id for event in resumed.delivered_events]
    assert len(event_ids) == len(set(event_ids))
    resumed_delivered_ids = {
        event.event_id
        for frame in actual_frames
        for event in frame.delivered_events
    }
    assert {event.event_id for event in first.generated_events}.issubset(
        resumed_delivered_ids
    )


def test_two_phase_interval_prevents_future_phase_and_same_interval_events() -> None:
    bridge = StreamingNOD1MotorBridge(_unit_gain_config(), seed=5)
    interval = bridge.begin_interval(
        circuit_sample=_sample(
            0,
            0.0,
            0.0,
            raw_l_voltage_v=-0.055,
            raw_r_voltage_v=-0.055,
        )
    )
    assert isinstance(interval, StreamingBridgeIntervalStart)
    assert interval.tick_index == 0
    assert interval.delivered_events == ()
    assert bridge.interval_is_open
    with pytest.raises(RuntimeError, match="already open"):
        bridge.begin_interval()
    with pytest.raises(RuntimeError, match="only between intervals"):
        bridge.checkpoint()

    frame = bridge.end_interval(0.90 * _TWO_PI)
    assert not bridge.interval_is_open
    assert len(frame.generated_events) == 8
    assert frame.delivered_events == ()
    assert all(
        event.availability_time_s >= frame.interval_end_s
        for event in frame.generated_events
    )

    # Only the next interval start can expose those earlier generated events to
    # mechanics, and their continuous NMJ availability times are preserved.
    next_interval = bridge.begin_interval()
    assert {event.event_id for event in next_interval.delivered_events} == {
        event.event_id for event in frame.generated_events
    }
    assert all(
        next_interval.interval_start_s
        <= event.availability_time_s
        < next_interval.interval_end_s
        for event in next_interval.delivered_events
    )
    bridge.end_interval(0.90 * _TWO_PI)


def test_open_interval_rejects_restore_and_circuit_push_without_mutation() -> None:
    source = StreamingNOD1MotorBridge(seed=13)
    checkpoint = source.checkpoint()
    source.begin_interval(
        circuit_sample=_sample(0, 0.0, 0.0)
    )
    with pytest.raises(RuntimeError, match="only between intervals"):
        source.restore(checkpoint)
    with pytest.raises(RuntimeError, match="only between intervals"):
        source.push_circuit_sample(_sample(1, 0.005, 0.005))
    source.end_interval(0.0)


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


def test_corrupt_or_incompatible_checkpoints_are_rejected_transactionally() -> None:
    intervention = StreamingBridgeIntervention(
        intervention_id="control",
        target_stage=StreamingBridgeStage.DN,
        target_name="DNp26",
        raw_app_side=RawAppSide.R,
        mode=StreamingInterventionMode.SCALE,
        start_s=0.0,
        end_s=0.005,
        magnitude=1.0,
    )
    source = StreamingNOD1MotorBridge(seed=3, interventions=(intervention,))
    source.step(
        0.1,
        circuit_sample=_sample(
            0, 0.0, 0.0, raw_l_voltage_v=-0.055, raw_r_voltage_v=-0.060
        ),
    )
    checkpoint = source.checkpoint()

    bad_digest = checkpoint.to_dict()
    bad_digest["state"]["tick_index"] = 999
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        StreamingBridgeCheckpoint(bad_digest)

    # A malicious producer can recompute a checksum, so restore also validates
    # semantic state. The active schedule below is inconsistent with tick 1.
    bad_state = checkpoint.to_dict()
    bad_state["state"]["active_intervention_ids"] = []
    bad_state["payload_sha256"] = _canonical_digest(bad_state)
    syntactically_valid = StreamingBridgeCheckpoint(bad_state)
    target = StreamingNOD1MotorBridge(seed=3, interventions=(intervention,))
    before = target.checkpoint().to_dict()
    with pytest.raises(ValueError, match="active intervention state mismatch"):
        target.restore(syntactically_valid)
    assert target.checkpoint().to_dict() == before

    with pytest.raises(ValueError, match="seed mismatch"):
        StreamingNOD1MotorBridge(seed=4, interventions=(intervention,)).restore(
            checkpoint
        )
    with pytest.raises(ValueError, match="intervention state mismatch"):
        StreamingNOD1MotorBridge(seed=3).restore(checkpoint)


def test_checkpoint_rejects_rehashed_nmj_delay_and_phase_history_tampering() -> None:
    config = _unit_gain_config(nmj_delay_s=0.002)
    source = StreamingNOD1MotorBridge(config, seed=23)
    source.step(
        0.21 * _TWO_PI,
        circuit_sample=_sample(
            0,
            0.0,
            0.0,
            raw_l_voltage_v=-0.055,
            raw_r_voltage_v=-0.055,
        ),
    )
    checkpoint = source.checkpoint()
    assert checkpoint.to_dict()["state"]["event_pending"]

    bad_delay = checkpoint.to_dict()
    queued = bad_delay["state"]["event_pending"][0]
    queued["value"]["availability_time_s"] += 0.0005
    queued["availability_time_s"] += 0.0005
    bad_delay["payload_sha256"] = _canonical_digest(bad_delay)
    target = StreamingNOD1MotorBridge(config, seed=23)
    before = target.checkpoint().to_dict()
    with pytest.raises(ValueError, match="provenance is inconsistent"):
        target.restore(StreamingBridgeCheckpoint(bad_delay))
    assert target.checkpoint().to_dict() == before

    bad_phase = checkpoint.to_dict()
    bad_phase["state"]["wing_phase_unwrapped_rad"] = 0.0
    bad_phase["payload_sha256"] = _canonical_digest(bad_phase)
    with pytest.raises(ValueError, match="path history"):
        target.restore(StreamingBridgeCheckpoint(bad_phase))
    assert target.checkpoint().to_dict() == before

    bad_crossing = checkpoint.to_dict()
    crossing_key = next(
        key
        for key, value in bad_crossing["state"]["last_crossing_time_s"].items()
        if value is not None
    )
    bad_crossing["state"]["last_crossing_time_s"][crossing_key] += 1e-5
    bad_crossing["payload_sha256"] = _canonical_digest(bad_crossing)
    with pytest.raises(ValueError, match="last crossing time is inconsistent"):
        target.restore(StreamingBridgeCheckpoint(bad_crossing))
    assert target.checkpoint().to_dict() == before

    bad_encoder_delay = checkpoint.to_dict()
    bad_encoder_delay["state"]["dn_hold"]["L"]["availability_time_s"] += 0.0005
    bad_encoder_delay["payload_sha256"] = _canonical_digest(bad_encoder_delay)
    with pytest.raises(ValueError, match="dn_hold channel identity mismatch"):
        target.restore(StreamingBridgeCheckpoint(bad_encoder_delay))
    assert target.checkpoint().to_dict() == before


def test_six_point_phase_path_locates_crossings_piecewise_not_linearly() -> None:
    bridge = StreamingNOD1MotorBridge(_unit_gain_config(), seed=29)
    start = bridge.begin_interval(
        circuit_sample=_sample(
            0,
            0.0,
            0.0,
            raw_l_voltage_v=-0.055,
            raw_r_voltage_v=-0.055,
        )
    )
    path = (0.0, 0.01, 0.02, 0.03, 0.04, 0.90 * _TWO_PI)
    frame = bridge.end_interval(path)
    assert frame.wing_phase_path_unwrapped_rad == path
    iv2 = [event for event in frame.generated_events if event.muscle == "iv2"]
    assert len(iv2) == 2
    preferred = 0.20 * _TWO_PI
    expected = 0.0004 + (preferred - 0.04) / (0.90 * _TWO_PI - 0.04) * 0.0001
    for event in iv2:
        assert event.event_time_s == pytest.approx(expected, abs=1e-15)
        assert event.event_time_s != pytest.approx(
            start.interval_start_s + (0.20 / 0.90) * 0.0005,
            abs=1e-10,
        )


def _batch_trace(samples) -> CircuitOutputTrace:
    dataset = DatasetRef(
        namespace=DatasetNamespace.FLYWIRE_FAFB,
        release="FAFB",
        materialization=783,
        source_uri="urn:test:streaming-bridge",
        coordinate_units="m",
    )
    provenance = Provenance(
        source_uri="urn:test:streaming-bridge",
        method="four-channel online/batch parity fixture",
        dataset_identity=dataset.identity_space,
    )
    confidence = Confidence(
        tier=EvidenceTier.MODEL_INFERENCE,
        level=ConfidenceLevel.LOW,
        score=0.2,
        basis="test-only raw application side convention",
    )
    signals = []
    for root_id in FLY_FGS_NOD1_ROOT_IDS:
        raw_side = FLY_FGS_NOD1_RAW_APP_SIDES[root_id]
        app_side = (
            AnatomicalSide.LEFT if raw_side == "L" else AnatomicalSide.RIGHT
        )
        side_context = SideContext(
            dataset=dataset,
            raw_dataset_side=raw_side,
            anatomical_side=AnatomicalSide.UNKNOWN,
            visual_field_side=AnatomicalSide.UNKNOWN,
            app_rendering_side=app_side,
            eye_side=EyeSide.UNKNOWN,
            effector_side=AnatomicalSide.UNKNOWN,
            mapping_method=SideMappingMethod.SIMULATION_CONVENTION,
            confidence=confidence,
            notes="raw fly-FGS application label; anatomy unknown",
        )
        signals.append(
            CircuitSignal(
                neuron=EntityRef(
                    dataset,
                    root_id,
                    EntityKind.NEURON,
                    cell_type="NOD1",
                    anatomical_side=AnatomicalSide.UNKNOWN,
                    side_context=side_context,
                ),
                signal_kind=CircuitSignalKind.VOLTAGE,
                unit="V",
                values=tuple(sample.nod1_voltage_v[root_id] for sample in samples),
                availability_times_s=tuple(
                    sample.availability_time_s for sample in samples
                ),
                provenance=provenance,
                confidence=confidence,
            )
        )
    return CircuitOutputTrace(
        dataset=dataset,
        sample_times_s=tuple(sample.measurement_time_s for sample in samples),
        signals=tuple(signals),
        provenance=provenance,
        confidence=confidence,
        exact_timebase=True,
        sample_interval_end_s=0.010,
    )


def test_streamed_rate_states_match_the_existing_batch_bridge() -> None:
    samples = (
        _sample(
            0, 0.0, 0.0, raw_l_voltage_v=-0.055, raw_r_voltage_v=-0.058
        ),
        _sample(
            1, 0.005, 0.005, raw_l_voltage_v=-0.057, raw_r_voltage_v=-0.060
        ),
    )
    config = StreamingBridgeConfig(
        encoder_delay_s=0.001,
        vnc_delay_s=0.001,
        nmj_delay_s=0.0005,
    )
    online = StreamingNOD1MotorBridge(config, seed=41)
    for sample in samples:
        online.push_circuit_sample(sample)
    frames = [online.step(0.0) for _ in range(20)]

    batch_dn = CircuitToDNEncoder(
        CircuitToDNEncoderConfig(
            update_dt_s=0.0005,
            input_availability_delay_s=0.0,
            encoder_delay_s=0.001,
            resting_voltage_v=config.resting_voltage_v,
            voltage_scale_v=config.voltage_scale_v,
            dn_baseline_rate_hz=config.dn_baseline_rate_hz,
            functional_gain_hz=config.dn_functional_gain_hz,
            maximum_dn_rate_hz=config.maximum_dn_rate_hz,
        )
    ).encode(_batch_trace(samples))
    batch_motor = VNCMotorNetwork(
        VNCMotorNetworkConfig(
            update_dt_s=0.0005,
            vnc_delay_s=0.001,
            maximum_motor_rate_hz=config.maximum_motor_rate_hz,
        )
    ).run(batch_dn, seed=41)

    side_map = {
        RawAppSide.L: AnatomicalSide.LEFT,
        RawAppSide.R: AnatomicalSide.RIGHT,
    }
    for side, anatomical_side in side_map.items():
        expected_dn = batch_dn.channel("DNp26", anatomical_side).samples
        actual_dn = [
            next(
                sample
                for sample in frame.generated_dn_rates
                if sample.raw_app_side is side
            )
            for frame in frames
        ]
        assert [sample.rate_hz for sample in actual_dn] == pytest.approx(
            [sample.rate_hz for sample in expected_dn]
        )
        assert [sample.availability_time_s for sample in actual_dn] == pytest.approx(
            [sample.availability_time_s for sample in expected_dn]
        )

        for path in config.pathways:
            expected_motor = batch_motor.channel(path.muscle, anatomical_side).rate_samples
            actual_motor = [
                next(
                    sample
                    for sample in frame.generated_motor_rates
                    if sample.raw_app_side is side
                    and sample.target_name == path.motor_neuron
                )
                for frame in frames
            ]
            assert [sample.rate_hz for sample in actual_motor] == pytest.approx(
                [sample.rate_hz for sample in expected_motor]
            )
            assert [
                sample.availability_time_s for sample in actual_motor
            ] == pytest.approx(
                [sample.availability_time_s for sample in expected_motor]
            )


def test_strict_grid_and_phase_contracts_reject_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="exact 0.5 ms"):
        StreamingBridgeConfig(base_dt_s=0.001)
    with pytest.raises(ValueError, match="0.5 ms grid"):
        StreamingBridgeConfig(encoder_delay_s=0.0006)
    with pytest.raises(ValueError, match="at least one 0.5 ms"):
        StreamingBridgeConfig(nmj_delay_s=0.0)

    bridge = StreamingNOD1MotorBridge()
    with pytest.raises(ValueError, match="cannot decrease"):
        bridge.step(-0.1)
    with pytest.raises(ValueError, match="more than one cycle"):
        bridge.step(_TWO_PI + 0.1)


def test_canonical_circuit_sample_inventory_and_exact_cadence_are_enforced() -> None:
    with pytest.raises(ValueError, match="begin at sample 0"):
        StreamingNOD1MotorBridge().push_circuit_sample(
            _sample(1, 0.005, 0.005)
        )

    bridge = StreamingNOD1MotorBridge()
    bridge.push_circuit_sample(_sample(0, 0.0, 0.0))
    with pytest.raises(ValueError, match="contiguous"):
        bridge.push_circuit_sample(_sample(2, 0.010, 0.010))

    with pytest.raises(ValueError, match="measurement_time_s"):
        StreamingNOD1MotorBridge().push_circuit_sample(
            _sample(0, 0.0005, 0.0005)
        )
    with pytest.raises(ValueError, match="availability_time_s"):
        StreamingNOD1MotorBridge().push_circuit_sample(
            _sample(0, 0.0, 0.0005)
        )
    with pytest.raises(ValueError, match=r"\[0, 99\]"):
        StreamingNOD1MotorBridge().push_circuit_sample(
            _sample(100, 0.5, 0.5)
        )
