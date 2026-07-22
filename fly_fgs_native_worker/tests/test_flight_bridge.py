import math
from dataclasses import replace

import pytest

from fly_sensor2behavior.flight.bridge import (
    AvailabilityQueue,
    AvailableValue,
    BridgeIntervention,
    BridgeInterventionMode,
    BridgeSignalSemantics,
    BridgeTargetType,
    CausalNeuralBridge,
    CircuitToDNEncoder,
    CircuitToDNEncoderConfig,
    ExactMotorEvent,
    MotorPathway,
    VNCMotorNetwork,
    VNCMotorNetworkConfig,
    default_steering_pathways,
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
    MotorSignalKind,
    Provenance,
    SideContext,
    SideMappingMethod,
)


def _dataset():
    return DatasetRef(
        namespace=DatasetNamespace.FLYWIRE_FAFB,
        release="FAFB",
        materialization=783,
        source_uri="http://54.160.228.98/drosophila/api/manifest",
        coordinate_units="m",
    )


def _provenance(dataset):
    return Provenance(
        source_uri="urn:test:nod1-bridge",
        method="deterministic bridge fixture",
        dataset_identity=dataset.identity_space,
    )


def _confidence():
    return Confidence(
        tier=EvidenceTier.MODEL_INFERENCE,
        level=ConfidenceLevel.LOW,
        score=0.2,
        basis="test-only exploratory fixture",
    )


def _trace(
    *,
    duration_s=0.020,
    source_dt_s=0.005,
    left_voltage_v=-0.055,
    right_voltage_v=-0.060,
    left_values=None,
    right_values=None,
    sample_interval_end_s=None,
):
    dataset = _dataset()
    count = int(round(duration_s / source_dt_s)) + 1
    times = tuple(index * source_dt_s for index in range(count))
    left = tuple(left_values) if left_values is not None else (left_voltage_v,) * count
    right = tuple(right_values) if right_values is not None else (right_voltage_v,) * count
    assert len(left) == count
    assert len(right) == count

    signals = []
    for index, (side, values) in enumerate(
        ((AnatomicalSide.LEFT, left), (AnatomicalSide.RIGHT, right))
    ):
        # Two individual cells per side exercise population averaging without
        # collapsing their source identities.
        for replicate in range(2):
            signals.append(
                CircuitSignal(
                    neuron=EntityRef(
                        dataset,
                        str(720575940600000000 + 10 * index + replicate),
                        EntityKind.NEURON,
                        cell_type="NOD1",
                        anatomical_side=side,
                    ),
                    signal_kind=CircuitSignalKind.VOLTAGE,
                    unit="V",
                    values=values,
                    provenance=_provenance(dataset),
                    confidence=_confidence(),
                )
            )
    return CircuitOutputTrace(
        dataset=dataset,
        sample_times_s=times,
        signals=tuple(signals),
        provenance=_provenance(dataset),
        confidence=_confidence(),
        exact_timebase=True,
        sample_interval_end_s=sample_interval_end_s,
    )


def _with_low_confidence_app_sides(trace):
    signals = []
    for signal in trace.signals:
        app_side = signal.neuron.anatomical_side
        context = SideContext(
            dataset=trace.dataset,
            raw_dataset_side="L" if app_side is AnatomicalSide.LEFT else "R",
            anatomical_side=AnatomicalSide.UNKNOWN,
            visual_field_side=AnatomicalSide.UNKNOWN,
            app_rendering_side=app_side,
            eye_side=EyeSide.UNKNOWN,
            effector_side=AnatomicalSide.UNKNOWN,
            mapping_method=SideMappingMethod.SIMULATION_CONVENTION,
            confidence=_confidence(),
            notes="test-only app convention; not anatomical laterality",
        )
        neuron = replace(
            signal.neuron,
            anatomical_side=AnatomicalSide.UNKNOWN,
            side_context=context,
        )
        signals.append(replace(signal, neuron=neuron))
    return replace(trace, signals=tuple(signals))


def _encode(trace=None, **config_overrides):
    values = {
        "update_dt_s": 0.001,
        "input_availability_delay_s": 0.0,
        "encoder_delay_s": 0.003,
    }
    values.update(config_overrides)
    return CircuitToDNEncoder(CircuitToDNEncoderConfig(**values)).encode(
        _trace() if trace is None else trace
    )


def _motor(descending, *, seed=7, interventions=(), exact_motor_events=(), **overrides):
    values = {"update_dt_s": 0.001, "vnc_delay_s": 0.002}
    values.update(overrides)
    return VNCMotorNetwork(VNCMotorNetworkConfig(**values)).run(
        descending,
        seed=seed,
        interventions=interventions,
        exact_motor_events=exact_motor_events,
    )


def test_availability_queue_never_releases_future_data_and_holds_last_value():
    queue = AvailabilityQueue()
    queue.push(AvailableValue(0.000, 0.003, "first"))
    queue.push(AvailableValue(0.002, 0.005, "second"))

    assert queue.advance(0.002) is None
    assert queue.advance(0.003).value == "first"
    assert queue.advance(0.004).value == "first"
    assert queue.advance(0.005).value == "second"
    with pytest.raises(ValueError, match="retroactively"):
        queue.push(AvailableValue(0.001, 0.004, "late insertion"))


def test_encoder_uses_zero_order_hold_and_has_exact_update_count():
    trace = _trace(
        duration_s=0.010,
        left_values=(-0.055, -0.060, -0.060),
        right_values=(-0.060, -0.060, -0.060),
    )
    encoder = CircuitToDNEncoder(
        CircuitToDNEncoderConfig(
            update_dt_s=0.001,
            input_availability_delay_s=0.002,
            encoder_delay_s=0.003,
        )
    )
    result = encoder.encode(trace)
    right_dn = result.channel("DNp26", AnatomicalSide.RIGHT)

    assert result.update_count == 10
    assert len(right_dn.samples) == 10
    assert [sample.rate_hz for sample in right_dn.samples[:2]] == [0.0, 0.0]
    assert [sample.rate_hz for sample in right_dn.samples[2:7]] == pytest.approx(
        [40.0] * 5
    )
    assert right_dn.samples[2].source_measurement_time_s == 0.0
    assert right_dn.samples[2].availability_time_s == pytest.approx(0.005)
    assert right_dn.samples[7].rate_hz == 0.0


def test_encoder_respects_declared_sample_availability_plus_bridge_delay():
    trace = _trace(
        duration_s=0.010,
        left_values=(-0.055, -0.060, -0.060),
        right_values=(-0.060, -0.060, -0.060),
    )
    trace = replace(
        trace,
        signals=tuple(
            replace(
                signal,
                availability_times_s=(0.004, 0.009, 0.014),
            )
            for signal in trace.signals
        ),
        sample_interval_end_s=0.015,
    )
    result = _encode(trace, input_availability_delay_s=0.002)
    right_dn = result.channel("DNp26", AnatomicalSide.RIGHT)

    assert all(sample.rate_hz == 0.0 for sample in right_dn.samples[:6])
    assert right_dn.samples[6].rate_hz == pytest.approx(40.0)
    assert right_dn.samples[6].source_measurement_time_s == pytest.approx(0.0)
    assert right_dn.samples[6].availability_time_s == pytest.approx(0.009)
    assert right_dn.samples[11].rate_hz == 0.0
    assert right_dn.samples[11].source_measurement_time_s == pytest.approx(0.005)


def test_encoder_respects_declared_spike_availability_plus_bridge_delay():
    trace = _trace(duration_s=0.010)
    trace = replace(
        trace,
        signals=tuple(
            replace(
                signal,
                signal_kind=CircuitSignalKind.SPIKE_EVENTS,
                unit="1",
                values=(),
                spike_times_s=(0.001,),
                availability_times_s=(0.006,),
            )
            for signal in trace.signals
        ),
    )
    result = _encode(trace, input_availability_delay_s=0.002)
    right_dn = result.channel("DNp26", AnatomicalSide.RIGHT)

    assert all(sample.rate_hz == 0.0 for sample in right_dn.samples[:8])
    assert right_dn.samples[8].rate_hz == pytest.approx(200.0)
    assert right_dn.samples[8].source_measurement_time_s == pytest.approx(0.001)
    assert right_dn.samples[8].availability_time_s == pytest.approx(0.011)


def test_encoder_is_contralateral_and_mirrors_rates_without_atlas_id_join():
    left_driven = _encode(_trace(left_voltage_v=-0.055, right_voltage_v=-0.060))
    right_driven = _encode(_trace(left_voltage_v=-0.060, right_voltage_v=-0.055))

    left_to_right = left_driven.channel("DNp26", AnatomicalSide.RIGHT)
    right_to_left = right_driven.channel("DNp26", AnatomicalSide.LEFT)
    assert [sample.rate_hz for sample in left_to_right.samples] == [
        sample.rate_hz for sample in right_to_left.samples
    ]
    assert all(
        sample.rate_hz == 0.0
        for sample in left_driven.channel("DNp26", AnatomicalSide.LEFT).samples
    )
    assert all("flywire_fafb:FAFB@783" in scoped for scoped in left_to_right.source_scoped_ids)
    assert "no cross-atlas identifier join" in left_to_right.mapping_method


def test_encoder_consumes_final_half_open_sample_and_preserves_latency_tail():
    trace = _trace(
        duration_s=0.010,
        left_values=(-0.060, -0.060, -0.050),
        right_values=(-0.060, -0.060, -0.060),
        sample_interval_end_s=0.015,
    )
    bridge = CausalNeuralBridge(
        encoder=CircuitToDNEncoder(
            CircuitToDNEncoderConfig(
                update_dt_s=0.001,
                input_availability_delay_s=0.0,
                encoder_delay_s=0.003,
            )
        ),
        motor_network=VNCMotorNetwork(
            VNCMotorNetworkConfig(update_dt_s=0.001, vnc_delay_s=0.002)
        ),
    ).run(trace, seed=5)

    assert bridge.descending.duration_s == pytest.approx(0.015)
    assert bridge.descending.update_count == 15
    right_dn = bridge.descending.channel("DNp26", AnatomicalSide.RIGHT)
    final_source_dn = next(
        sample
        for sample in right_dn.samples
        if sample.measurement_time_s == pytest.approx(0.010)
    )
    assert final_source_dn.source_measurement_time_s == pytest.approx(0.010)
    assert final_source_dn.rate_hz > 0.0
    assert final_source_dn.availability_time_s == pytest.approx(0.013)

    right_iv2 = bridge.wing_motor.channel("iv2", AnatomicalSide.RIGHT)
    latency_tail = next(
        sample
        for sample in right_iv2.rate_samples
        if sample.measurement_time_s == pytest.approx(0.013)
    )
    assert latency_tail.source_measurement_time_s == pytest.approx(0.010)
    assert latency_tail.rate_hz > 0.0
    assert latency_tail.availability_time_s == pytest.approx(0.015)


def test_encoder_uses_explicit_app_convention_without_claiming_anatomical_side():
    trace = _with_low_confidence_app_sides(
        _trace(left_voltage_v=-0.055, right_voltage_v=-0.060)
    )
    assert all(
        signal.neuron.anatomical_side is AnatomicalSide.UNKNOWN
        for signal in trace.signals
    )

    descending = _encode(trace)
    right_dn = descending.channel("DNp26", AnatomicalSide.RIGHT)
    assert any(sample.rate_hz > 0.0 for sample in right_dn.samples)
    assert "not anatomical laterality" in right_dn.mapping_method

    missing_context = replace(
        trace,
        signals=tuple(
            replace(
                signal,
                neuron=replace(signal.neuron, side_context=None),
            )
            for signal in trace.signals
        ),
    )
    with pytest.raises(ValueError, match="explicit low-confidence"):
        _encode(missing_context)


def test_combined_delays_prevent_motor_effect_before_availability():
    descending = _encode(
        _trace(duration_s=0.050, left_voltage_v=-0.055),
        encoder_delay_s=0.003,
    )
    motor = _motor(descending, vnc_delay_s=0.002, seed=4)
    iv2_right = motor.channel("iv2", AnatomicalSide.RIGHT)

    assert motor.update_count == 50
    # The first DN rate is available at 3 ms. The motor update at 3 ms is
    # itself available at 5 ms, and no synthetic event may precede that bound.
    assert all(sample.rate_hz == 0.0 for sample in iv2_right.rate_samples[:3])
    assert iv2_right.rate_samples[3].rate_hz > 0.0
    assert iv2_right.rate_samples[3].availability_time_s == pytest.approx(0.005)
    assert all(event.event_time_s >= 0.005 for event in iv2_right.events)
    assert all(
        event.semantics is BridgeSignalSemantics.SEEDED_SYNTHETIC
        for event in iv2_right.events
    )


def test_seeded_phase_events_are_reproducible_and_seed_sensitive():
    trace = _trace(duration_s=0.100, left_voltage_v=-0.050)
    descending = _encode(trace, functional_gain_hz=100.0)
    first = _motor(descending, seed=19)
    repeated = _motor(descending, seed=19)
    changed = _motor(descending, seed=20)

    first_events = first.channel("iv2", AnatomicalSide.RIGHT).events
    repeated_events = repeated.channel("iv2", AnatomicalSide.RIGHT).events
    changed_events = changed.channel("iv2", AnatomicalSide.RIGHT).events
    assert first_events == repeated_events
    assert first_events != changed_events
    assert first_events
    assert all(
        event.wingbeat_phase_rad
        == pytest.approx(first.channel("iv2", AnatomicalSide.RIGHT).preferred_phase_rad)
        for event in first_events
    )


def test_exact_motor_events_remain_exact_and_are_not_mixed_with_inferred_rate():
    descending = _encode(_trace(duration_s=0.020))
    exact = ExactMotorEvent(
        motor_neuron="MN-iv2",
        muscle="iv2",
        side=AnatomicalSide.RIGHT,
        event_time_s=0.007,
        availability_time_s=0.008,
        wingbeat_phase_rad=0.4,
    )
    motor = _motor(descending, exact_motor_events=(exact,))
    channel = motor.channel("iv2", AnatomicalSide.RIGHT)

    assert channel.rate_samples == ()
    assert len(channel.events) == 1
    assert channel.events[0].semantics is BridgeSignalSemantics.EXACT_SPIKES
    assert channel.events[0].generator_seed is None
    schema_signal = next(
        signal
        for signal in motor.to_schema_trace().signals
        if signal.muscle.cell_type == "iv2"
        and signal.side is AnatomicalSide.RIGHT
    )
    assert schema_signal.signal_kind is MotorSignalKind.EXACT_SPIKES
    assert schema_signal.spike_times_s == (0.007,)
    assert schema_signal.availability_times_s == (0.008,)
    runtime_command = next(
        command
        for command in motor.to_flight_motor_commands()
        if command.muscle == "iv2" and command.side.value == "right"
    )
    assert runtime_command.measurement_spike_times_s == (0.007,)
    assert runtime_command.spike_times_s == (0.008,)


def test_schema_export_keeps_synthetic_events_labelled_as_inferred_rates():
    motor = _motor(_encode(_trace(duration_s=0.050)), seed=1)
    bridge_channel = motor.channel("iv2", AnatomicalSide.RIGHT)
    assert bridge_channel.event_semantics is BridgeSignalSemantics.SEEDED_SYNTHETIC

    schema_trace = motor.to_schema_trace()
    schema_signal = next(
        signal
        for signal in schema_trace.signals
        if signal.muscle.cell_type == "iv2"
        and signal.side is AnatomicalSide.RIGHT
    )
    assert schema_signal.signal_kind is MotorSignalKind.INFERRED_RATE
    assert schema_signal.spike_times_s == ()
    assert schema_signal.availability_times_s == tuple(
        sample.availability_time_s for sample in bridge_channel.rate_samples
    )
    assert schema_trace.dataset.namespace is DatasetNamespace.SIMULATION
    assert all(
        signal.motor_neuron.dataset.identity_space == signal.muscle.dataset.identity_space
        for signal in schema_trace.signals
    )


def test_muscle_intervention_is_local_to_one_named_side_and_random_stream():
    descending = _encode(
        _trace(duration_s=0.100, left_voltage_v=-0.050, right_voltage_v=-0.050),
        functional_gain_hz=100.0,
    )
    baseline = _motor(descending, seed=31)
    silence_b3_right = BridgeIntervention(
        target_type=BridgeTargetType.MUSCLE,
        target_name="b3",
        side=AnatomicalSide.RIGHT,
        mode=BridgeInterventionMode.SILENCE,
        start_s=0.0,
        end_s=0.100,
    )
    silenced = _motor(descending, seed=31, interventions=(silence_b3_right,))

    assert all(
        sample.rate_hz == 0.0
        for sample in silenced.channel("b3", AnatomicalSide.RIGHT).rate_samples
    )
    assert silenced.channel("b3", AnatomicalSide.RIGHT).events == ()
    assert (
        silenced.channel("b3", AnatomicalSide.LEFT)
        == baseline.channel("b3", AnatomicalSide.LEFT)
    )
    assert (
        silenced.channel("iv2", AnatomicalSide.RIGHT)
        == baseline.channel("iv2", AnatomicalSide.RIGHT)
    )


def test_structural_count_magnitude_is_not_used_as_functional_gain():
    descending = _encode(_trace(duration_s=0.020))
    low_count = MotorPathway("MN-iv2", "iv2", 0.6, 0.2, True, 1)
    high_count = MotorPathway("MN-iv2", "iv2", 0.6, 0.2, True, 10_000)
    low = _motor(descending, pathways=(low_count,), seed=9)
    high = _motor(descending, pathways=(high_count,), seed=9)

    assert [sample.rate_hz for sample in low.channel("iv2", AnatomicalSide.RIGHT).rate_samples] == [
        sample.rate_hz for sample in high.channel("iv2", AnatomicalSide.RIGHT).rate_samples
    ]
    assert low.channel("iv2", AnatomicalSide.RIGHT).events == high.channel(
        "iv2", AnatomicalSide.RIGHT
    ).events


def test_mn_side_mirror_preserves_individual_muscle_rate_pattern():
    left_driven = _motor(
        _encode(_trace(left_voltage_v=-0.055, right_voltage_v=-0.060)), seed=2
    )
    right_driven = _motor(
        _encode(_trace(left_voltage_v=-0.060, right_voltage_v=-0.055)), seed=2
    )
    for pathway in default_steering_pathways():
        right_rates = [
            sample.rate_hz
            for sample in left_driven.channel(pathway.muscle, AnatomicalSide.RIGHT).rate_samples
        ]
        mirrored_left_rates = [
            sample.rate_hz
            for sample in right_driven.channel(pathway.muscle, AnatomicalSide.LEFT).rate_samples
        ]
        assert right_rates == mirrored_left_rates
        assert any(rate > 0.0 for rate in right_rates)


def test_end_to_end_facade_routes_dn_intervention_exactly_once():
    scale_dn = BridgeIntervention(
        target_type=BridgeTargetType.DN,
        target_name="DNp26",
        side=AnatomicalSide.RIGHT,
        mode=BridgeInterventionMode.SCALE,
        start_s=0.0,
        end_s=0.020,
        magnitude=0.5,
    )
    bridge = CausalNeuralBridge(
        encoder=CircuitToDNEncoder(
            CircuitToDNEncoderConfig(update_dt_s=0.001, encoder_delay_s=0.003)
        ),
        motor_network=VNCMotorNetwork(
            VNCMotorNetworkConfig(update_dt_s=0.001, vnc_delay_s=0.002)
        ),
    )
    result = bridge.run(_trace(), seed=5, interventions=(scale_dn,))

    right_dn = result.descending.channel("DNp26", AnatomicalSide.RIGHT)
    assert right_dn.samples[0].rate_hz == pytest.approx(20.0)
    # iv2's independent functional gain is 0.6, so a single 0.5 DN scale
    # produces 40 * 0.5 * 0.6 = 12 Hz once the delayed DN value is available.
    iv2_rates = result.wing_motor.channel(
        "iv2", AnatomicalSide.RIGHT
    ).rate_samples
    assert iv2_rates[3].rate_hz == pytest.approx(12.0)


def test_exact_event_non_silencing_intervention_is_rejected():
    descending = _encode(_trace(duration_s=0.020))
    exact = ExactMotorEvent(
        motor_neuron="MN-iv2",
        muscle="iv2",
        side=AnatomicalSide.RIGHT,
        event_time_s=0.007,
    )
    scale = BridgeIntervention(
        target_type=BridgeTargetType.MN,
        target_name="MN-iv2",
        side=AnatomicalSide.RIGHT,
        mode=BridgeInterventionMode.SCALE,
        start_s=0.0,
        end_s=0.020,
        magnitude=0.5,
    )
    with pytest.raises(ValueError, match="exact motor events support silencing only"):
        _motor(descending, exact_motor_events=(exact,), interventions=(scale,))

    # A same-name intervention on the opposite anatomical side remains local
    # and must not alter or invalidate the exact right-side event.
    opposite_side_scale = BridgeIntervention(
        target_type=BridgeTargetType.MN,
        target_name="MN-iv2",
        side=AnatomicalSide.LEFT,
        mode=BridgeInterventionMode.SCALE,
        start_s=0.0,
        end_s=0.020,
        magnitude=0.5,
    )
    opposite = _motor(
        descending,
        exact_motor_events=(exact,),
        interventions=(opposite_side_scale,),
    )
    assert opposite.channel("iv2", AnatomicalSide.RIGHT).events[0].event_time_s == 0.007

    silence = BridgeIntervention(
        target_type=BridgeTargetType.MN,
        target_name="MN-iv2",
        side=AnatomicalSide.RIGHT,
        mode=BridgeInterventionMode.SILENCE,
        start_s=0.0,
        end_s=0.020,
    )
    silenced = _motor(
        descending,
        exact_motor_events=(exact,),
        interventions=(silence,),
    )
    assert silenced.channel("iv2", AnatomicalSide.RIGHT).events == ()
    exact_schema = next(
        signal
        for signal in silenced.to_schema_trace().signals
        if signal.muscle.cell_type == "iv2"
        and signal.side is AnatomicalSide.RIGHT
    )
    assert exact_schema.signal_kind is MotorSignalKind.EXACT_SPIKES
    assert exact_schema.spike_times_s == ()
