import hashlib
import json
import math
from pathlib import Path

import pytest

from fly_sensor2behavior.schema import (
    ActuationOwner,
    AerodynamicsOwner,
    AnatomicalSide,
    AtlasMappingKind,
    CircuitOutputTrace,
    CircuitSignal,
    CircuitSignalKind,
    Confidence,
    ConfidenceLevel,
    CoverageSummary,
    CrossAtlasJoinError,
    DatasetNamespace,
    DatasetRef,
    DescendingSignal,
    DescendingTrace,
    EncoderProvenance,
    EntityKind,
    EntityRef,
    EpisodeStatus,
    EvidenceTier,
    EyeSide,
    FeedbackFrame,
    FlightEpisodeConfig,
    FlightEpisodeResult,
    ModelParameter,
    ModelRecord,
    ModelRegistry,
    MechanicsBackend,
    MechanicsFrame,
    MotorSignal,
    MotorSignalKind,
    MuscleClass,
    MuscleSeries,
    MuscleState,
    PerturbationSpec,
    Provenance,
    RetinalFrame,
    RetinalSignalKind,
    SchemaValidationError,
    SideContext,
    SideMappingMethod,
    SignalOrigin,
    ValidationStatus,
    WingMotorTrace,
    fafb_anatomical_side_from_soma_x_nm,
    require_same_identifier_space,
)


def flywire_dataset():
    return DatasetRef(
        namespace=DatasetNamespace.FLYWIRE_FAFB,
        release="FAFB",
        materialization=783,
        source_uri="http://54.160.228.98/drosophila/api/manifest",
        neuron_universe="139255 proofread roots",
        coordinate_units="nm",
    )


def manc_dataset():
    return DatasetRef(
        namespace=DatasetNamespace.MANC,
        release="male-cns:v1.0",
        source_uri="http://54.160.228.98/descending/",
        coordinate_units="m",
    )


def banc_dataset():
    return DatasetRef(
        namespace=DatasetNamespace.BANC,
        release="v888",
        source_uri="urn:test:banc:v888",
        coordinate_units="m",
    )


def provenance(dataset=None):
    return Provenance(
        source_uri="urn:test:fixture",
        method="deterministic test fixture",
        dataset_identity=None if dataset is None else dataset.identity_space,
    )


def confidence(tier=EvidenceTier.DIRECT_OBSERVATION):
    return Confidence(
        tier=tier,
        level=ConfidenceLevel.HIGH,
        score=0.95,
        basis="test fixture with explicit assumptions",
    )


def test_flywire_ids_are_string_scoped_and_json_safe():
    dataset = flywire_dataset()
    neuron = EntityRef(
        dataset=dataset,
        entity_id="720575940627502338",
        kind=EntityKind.NEURON,
        cell_type="VCH",
    )

    payload = json.loads(neuron.to_json())
    assert payload["entity_id"] == "720575940627502338"
    assert isinstance(payload["entity_id"], str)
    assert dataset.identity_space == "flywire_fafb:FAFB@783"

    with pytest.raises(SchemaValidationError, match="must be a string"):
        EntityRef(dataset=dataset, entity_id=720575940627502338, kind=EntityKind.NEURON)
    with pytest.raises(SchemaValidationError, match="decimal strings"):
        EntityRef(dataset=dataset, entity_id="r720575940627502338", kind=EntityKind.NEURON)


def test_flywire_requires_materialization_and_non_flywire_forbids_it():
    with pytest.raises(SchemaValidationError, match="materialization"):
        DatasetRef(namespace="flywire_fafb", release="FAFB")
    with pytest.raises(SchemaValidationError, match="only valid"):
        DatasetRef(namespace="manc", release="male-cns:v1.0", materialization=783)


def test_fafb_corrected_laterality_is_position_based():
    assert fafb_anatomical_side_from_soma_x_nm(533_001.0) is AnatomicalSide.LEFT
    assert fafb_anatomical_side_from_soma_x_nm(532_999.0) is AnatomicalSide.RIGHT
    assert fafb_anatomical_side_from_soma_x_nm(533_000.0) is AnatomicalSide.MIDLINE


def test_side_context_preserves_raw_labels_and_all_coordinate_frames():
    dataset = flywire_dataset()
    side = SideContext(
        dataset=dataset,
        raw_dataset_side="right",
        anatomical_side="left",
        visual_field_side="right",
        app_rendering_side="left",
        eye_side="left",
        effector_side="right",
        mapping_method="fafb_soma_x",
        confidence=confidence(EvidenceTier.CURATED_CONNECTOME),
        soma_x_nm=534_000.0,
    )
    neuron = EntityRef(
        dataset=dataset,
        entity_id="720575940627502338",
        kind=EntityKind.NEURON,
        side_context=side,
    )
    assert neuron.anatomical_side is AnatomicalSide.LEFT
    assert neuron.side_context.raw_dataset_side == "right"
    assert EntityRef.from_dict(neuron.to_dict()) == neuron

    with pytest.raises(SchemaValidationError, match="corrected FAFB soma-x"):
        SideContext(
            **{**side.__dict__, "anatomical_side": "right"}
        )
    with pytest.raises(SchemaValidationError, match="not anatomical truth"):
        SideContext(
            **{
                **side.__dict__,
                "mapping_method": SideMappingMethod.CURATED_ATLAS_LABEL,
                "soma_x_nm": None,
            }
        )
    with pytest.raises(CrossAtlasJoinError, match="different atlas"):
        EntityRef(
            dataset=manc_dataset(),
            entity_id="10360",
            kind=EntityKind.NEURON,
            side_context=side,
        )


def test_direct_cross_atlas_identifier_join_is_rejected():
    flywire = EntityRef(flywire_dataset(), "720575940627502338", EntityKind.NEURON)
    manc = EntityRef(manc_dataset(), "10360", EntityKind.NEURON)
    with pytest.raises(CrossAtlasJoinError, match="explicit cell-type crosswalk"):
        require_same_identifier_space(flywire, manc)


def test_circuit_trace_requires_si_units_and_matching_timebase():
    dataset = flywire_dataset()
    signal = CircuitSignal(
        neuron=EntityRef(dataset, "720575940627502338", EntityKind.NEURON, cell_type="VCH"),
        signal_kind=CircuitSignalKind.VOLTAGE,
        unit="V",
        values=(-0.060, -0.055, -0.050),
        provenance=provenance(dataset),
        confidence=confidence(),
    )
    trace = CircuitOutputTrace(
        dataset=dataset,
        sample_times_s=(0.0, 0.005, 0.010),
        signals=(signal,),
        provenance=provenance(dataset),
        confidence=confidence(),
        exact_timebase=True,
        sample_interval_end_s=0.015,
    )
    assert trace.to_dict()["sample_times_s"] == [0.0, 0.005, 0.01]
    assert trace.to_dict()["sample_interval_end_s"] == pytest.approx(0.015)

    with pytest.raises(SchemaValidationError, match="supported SI unit"):
        CircuitSignal(
            neuron=signal.neuron,
            signal_kind="voltage",
            unit="mV",
            values=(-60.0,),
            provenance=provenance(dataset),
            confidence=confidence(),
        )
    with pytest.raises(SchemaValidationError, match="match sample_times_s"):
        CircuitOutputTrace(
            dataset=dataset,
            sample_times_s=(0.0, 0.005),
            signals=(signal,),
            provenance=provenance(dataset),
            confidence=confidence(),
            exact_timebase=True,
        )

    with pytest.raises(
        SchemaValidationError,
        match="greater than the final sample time",
    ):
        CircuitOutputTrace(
            dataset=dataset,
            sample_times_s=(0.0, 0.005, 0.010),
            signals=(signal,),
            provenance=provenance(dataset),
            confidence=confidence(),
            exact_timebase=True,
            sample_interval_end_s=0.010,
        )


def test_circuit_signal_availability_is_causal_and_backwards_compatible():
    dataset = flywire_dataset()
    neuron = EntityRef(dataset, "720575940627502338", EntityKind.NEURON)
    legacy_signal = CircuitSignal(
        neuron=neuron,
        signal_kind="voltage",
        unit="V",
        values=(-0.060, -0.055),
        provenance=provenance(dataset),
        confidence=confidence(),
    )
    assert legacy_signal.origin is SignalOrigin.UNSPECIFIED
    assert legacy_signal.availability_times_s == ()

    causal_signal = CircuitSignal(
        neuron=neuron,
        signal_kind="voltage",
        unit="V",
        values=(-0.060, -0.055),
        provenance=provenance(dataset),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
        origin="simulated",
        availability_times_s=(0.002, 0.007),
        source_signal_ids=("retina:left:0",),
    )
    trace = CircuitOutputTrace(
        dataset=dataset,
        sample_times_s=(0.0, 0.005),
        signals=(causal_signal,),
        provenance=provenance(dataset),
        confidence=confidence(),
        exact_timebase=True,
    )
    assert trace.signals[0].availability_times_s[-1] == pytest.approx(0.007)

    with pytest.raises(SchemaValidationError, match="before it is measured"):
        CircuitOutputTrace(
            dataset=dataset,
            sample_times_s=(0.0, 0.005),
            signals=(
                CircuitSignal(
                    **{
                        **causal_signal.__dict__,
                        "availability_times_s": (0.0, 0.004),
                    }
                ),
            ),
            provenance=provenance(dataset),
            confidence=confidence(),
            exact_timebase=True,
        )


def test_retinal_frame_enforces_exposure_causality_and_radiometric_units():
    frame = RetinalFrame(
        measurement_time_s=0.0015,
        availability_time_s=0.0022,
        exposure_start_s=0.001,
        exposure_end_s=0.002,
        eye_side=EyeSide.LEFT,
        signal_kind=RetinalSignalKind.NORMALIZED_LUMINANCE,
        unit="1",
        samples=(0.25, 0.75),
        ommatidial_directions_body=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        uncertainty=(0.01, 0.02),
        calibration_id="eye-calibration-v1",
        provenance=provenance(),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
    )
    assert RetinalFrame.from_dict(frame.to_dict()) == frame

    with pytest.raises(SchemaValidationError, match=r"unit W m\^-2"):
        RetinalFrame(
            **{
                **frame.__dict__,
                "signal_kind": RetinalSignalKind.IRRADIANCE,
                "unit": "1",
            }
        )
    with pytest.raises(SchemaValidationError, match="end of exposure"):
        RetinalFrame(**{**frame.__dict__, "availability_time_s": 0.0019})
    with pytest.raises(SchemaValidationError, match=r"\[0, 1\]"):
        RetinalFrame(**{**frame.__dict__, "samples": (0.5, 1.1)})


def test_motor_trace_distinguishes_exact_spikes_from_inferred_rates():
    dataset = manc_dataset()
    motor_neuron = EntityRef(
        dataset,
        "10360",
        EntityKind.MOTOR_NEURON,
        cell_type="MN_b2",
        anatomical_side="right",
    )
    muscle = EntityRef(dataset, "b2", EntityKind.MUSCLE, anatomical_side="right")
    exact = MotorSignal(
        motor_neuron=motor_neuron,
        muscle=muscle,
        side="right",
        signal_kind="exact_spikes",
        spike_times_s=(0.0011, 0.0062),
        wingbeat_phase_rad=(0.1, 0.2),
        provenance=provenance(dataset),
        confidence=confidence(),
    )
    trace = WingMotorTrace(
        dataset=dataset,
        duration_s=0.01,
        signals=(exact,),
        provenance=provenance(dataset),
        confidence=confidence(),
    )
    assert trace.signals[0].signal_kind is MotorSignalKind.EXACT_SPIKES
    with pytest.raises(SchemaValidationError, match="outside trace duration"):
        WingMotorTrace(
            dataset=dataset,
            duration_s=0.0062,
            signals=(exact,),
            provenance=provenance(dataset),
            confidence=confidence(),
        )

    inferred = MotorSignal(
        motor_neuron=motor_neuron,
        muscle=muscle,
        side="right",
        signal_kind="inferred_rate",
        sample_times_s=(0.0, 0.005),
        rate_hz=(20.0, 22.0),
        provenance=provenance(dataset),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
        origin=SignalOrigin.INFERRED,
        availability_times_s=(0.002, 0.007),
        source_signal_ids=("banc888:DNp26-left:rate",),
    )
    assert inferred.rate_hz == (20.0, 22.0)
    assert inferred.availability_times_s == (0.002, 0.007)

    with pytest.raises(SchemaValidationError, match="cannot also contain inferred rates"):
        MotorSignal(
            motor_neuron=motor_neuron,
            muscle=muscle,
            side="right",
            signal_kind="exact_spikes",
            spike_times_s=(0.001,),
            sample_times_s=(0.0,),
            rate_hz=(20.0,),
            provenance=provenance(dataset),
            confidence=confidence(),
        )
    with pytest.raises(SchemaValidationError, match="before it is measured"):
        MotorSignal(
            **{
                **inferred.__dict__,
                "availability_times_s": (0.0, 0.004),
            }
        )

    left_muscle = EntityRef(
        dataset, "b2-left", EntityKind.MUSCLE, anatomical_side="left"
    )
    with pytest.raises(SchemaValidationError, match="must match motor signal side"):
        MotorSignal(
            motor_neuron=motor_neuron,
            muscle=left_muscle,
            side="right",
            signal_kind="exact_spikes",
            spike_times_s=(0.001,),
            provenance=provenance(dataset),
            confidence=confidence(),
        )


def test_descending_trace_requires_explicit_encoder_bridge_and_causal_delays():
    input_dataset = flywire_dataset()
    output_dataset = banc_dataset()
    bridge_provenance = Provenance(
        source_uri="urn:test:banc-fafb-type-crosswalk",
        method="cell-type and anatomical-side crosswalk; never an ID join",
        derived_from=(input_dataset.identity_space, output_dataset.identity_space),
    )
    encoder = EncoderProvenance(
        name="nod1-to-dnp26",
        version="0.1.0",
        artifact_hash="sha256:encoder-fixture",
        input_dataset=input_dataset,
        output_dataset=output_dataset,
        mapping_kind=AtlasMappingKind.MODEL_BRIDGE,
        method="causal state-space fixture",
        provenance=provenance(),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
        cross_atlas_mapping=bridge_provenance,
    )
    dn = EntityRef(
        output_dataset,
        "DNp26-left",
        EntityKind.NEURON,
        cell_type="DNp26",
        anatomical_side="left",
    )
    common = {
        "descending_neuron": dn,
        "origin": SignalOrigin.INFERRED,
        "encoder_delay_s": 0.003,
        "availability_times_s": (0.003, 0.004, 0.005),
        "provenance": provenance(output_dataset),
        "confidence": confidence(EvidenceTier.MODEL_INFERENCE),
        "source_signal_ids": ("fafb783:NOD1-left:voltage",),
    }
    rate = DescendingSignal(
        **common,
        signal_kind="firing_rate",
        unit="Hz",
        values=(10.0, 12.0, 11.0),
    )
    voltage = DescendingSignal(
        **common,
        signal_kind="voltage",
        unit="V",
        values=(-0.060, -0.058, -0.059),
    )
    spikes = DescendingSignal(
        descending_neuron=dn,
        signal_kind="spike_events",
        unit="1",
        origin="synthetic",
        encoder_delay_s=0.003,
        spike_times_s=(0.001, 0.004),
        availability_times_s=(0.004, 0.007),
        provenance=provenance(output_dataset),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
        source_signal_ids=("banc888:DNp26-left:rate",),
    )
    trace = DescendingTrace(
        dataset=output_dataset,
        duration_s=0.01,
        sample_times_s=(0.0, 0.001, 0.002),
        signals=(voltage, rate, spikes),
        encoder=encoder,
        input_trace_hash="sha256:nod1-trace-fixture",
        provenance=provenance(output_dataset),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
    )
    restored = DescendingTrace.from_dict(trace.to_dict())
    assert restored == trace
    assert {signal.signal_kind for signal in trace.signals} == {
        CircuitSignalKind.VOLTAGE,
        CircuitSignalKind.FIRING_RATE,
        CircuitSignalKind.SPIKE_EVENTS,
    }

    with pytest.raises(CrossAtlasJoinError, match="explicit mapping provenance"):
        EncoderProvenance(
            **{**encoder.__dict__, "cross_atlas_mapping": None}
        )
    with pytest.raises(CrossAtlasJoinError, match="shared identifier space"):
        EncoderProvenance(
            **{
                **encoder.__dict__,
                "mapping_kind": AtlasMappingKind.SAME_IDENTIFIER_SPACE,
            }
        )
    with pytest.raises(SchemaValidationError, match="violates encoder_delay_s"):
        DescendingTrace(
            **{
                **trace.__dict__,
                "signals": (
                    DescendingSignal(
                        **{
                            **rate.__dict__,
                            "availability_times_s": (0.002, 0.004, 0.005),
                        }
                    ),
                ),
            }
        )
    with pytest.raises(SchemaValidationError, match="non-negative"):
        DescendingSignal(
            **{
                **rate.__dict__,
                "values": (10.0, -1.0, 11.0),
            }
        )


def test_muscle_state_carries_si_force_geometry_and_confidence():
    dataset = manc_dataset()
    series = MuscleSeries(
        muscle=EntityRef(dataset, "DLM", EntityKind.MUSCLE, anatomical_side="bilateral"),
        muscle_class=MuscleClass.ASYNCHRONOUS_POWER,
        activation=(0.1, 0.2),
        calcium_mol_m3=(0.01, 0.02),
        length_m=(1.0e-3, 1.01e-3),
        velocity_m_s=(0.0, 0.002),
        force_n=(1.0e-5, 1.2e-5),
        moment_arm_m=(1.0e-4,),
        provenance=provenance(dataset),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
    )
    state = MuscleState(
        dataset=dataset,
        sample_times_s=(0.0, 0.0001),
        muscles=(series,),
        provenance=provenance(dataset),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
    )
    assert state.muscles[0].force_n[-1] == pytest.approx(1.2e-5)

    stretch_amplified = MuscleSeries(
        **{**series.__dict__, "activation": (1.1, 1.5)}
    )
    assert stretch_amplified.activation[-1] == 1.5
    with pytest.raises(SchemaValidationError, match=r"\[0, 1\.5\]"):
        MuscleSeries(**{**series.__dict__, "activation": (0.1, 1.5001)})

    with pytest.raises(SchemaValidationError, match="moment arm or an attachment"):
        MuscleSeries(
            muscle=series.muscle,
            muscle_class="asynchronous_power",
            activation=(0.1,),
            calcium_mol_m3=(0.01,),
            length_m=(1e-3,),
            velocity_m_s=(0.0,),
            force_n=(1e-5,),
            provenance=provenance(dataset),
            confidence=confidence(),
        )


def test_mechanics_frame_has_exclusive_aero_actuation_and_si_wrenches():
    frame = MechanicsFrame(
        measurement_time_s=0.001,
        availability_time_s=0.0011,
        backend=MechanicsBackend.REDUCED_ANALYTIC,
        backend_version="analytic-flight-v1",
        aerodynamics_owner=AerodynamicsOwner.REDUCED_ANALYTIC,
        actuation_owner=ActuationOwner.NEUROMUSCULAR_ADAPTER,
        body_position_m=(0.0, 0.0, 0.01),
        body_orientation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        body_linear_velocity_m_s=(0.01, 0.0, 0.0),
        body_angular_velocity_rad_s=(0.0, 0.0, 0.1),
        wing_angles_rad={
            "left": (0.5, 0.1, -0.2),
            "right": (-0.5, 0.1, 0.2),
        },
        wing_angular_velocity_rad_s={
            "left": (100.0, 10.0, -20.0),
            "right": (-100.0, 10.0, 20.0),
        },
        aerodynamic_force_n=(0.0, 0.0, 1.0e-5),
        aerodynamic_moment_n_m=(0.0, 0.0, 1.0e-9),
        net_force_n=(0.0, 0.0, 1.9e-6),
        net_moment_n_m=(0.0, 0.0, 1.0e-9),
        kinetic_energy_j=1.0e-8,
        potential_energy_j=2.0e-7,
        actuator_work_j=2.0e-9,
        aerodynamic_work_j=-1.0e-10,
        provenance=provenance(),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
        numerical_diagnostics={"physics_timestep_s": 0.0001},
    )
    assert MechanicsFrame.from_dict(frame.to_dict()) == frame
    assert frame.mechanical_energy_j == pytest.approx(2.1e-7)

    with pytest.raises(SchemaValidationError, match="incompatible"):
        MechanicsFrame(
            **{
                **frame.__dict__,
                "aerodynamics_owner": AerodynamicsOwner.FLYBODY,
            }
        )
    with pytest.raises(SchemaValidationError, match="must be zero"):
        MechanicsFrame(
            **{
                **frame.__dict__,
                "aerodynamics_owner": AerodynamicsOwner.NONE,
            }
        )
    with pytest.raises(SchemaValidationError, match="before it is measured"):
        MechanicsFrame(**{**frame.__dict__, "availability_time_s": 0.0009})
    with pytest.raises(SchemaValidationError, match="exactly left and right"):
        MechanicsFrame(
            **{
                **frame.__dict__,
                "wing_angles_rad": {**frame.wing_angles_rad, "unknown": (0.0,)},
            }
        )


def test_feedback_frame_validates_quaternion_phase_and_latency():
    frame = FeedbackFrame(
        timestamp_s=0.001,
        retinal_irradiance_w_m2=(1.0, 0.5),
        body_position_m=(0.0, 0.0, 0.01),
        body_orientation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        body_linear_velocity_m_s=(0.0, 0.0, 0.0),
        body_angular_velocity_rad_s=(0.0, 0.0, 0.0),
        wing_phase_rad=(0.1, 0.1),
        wing_strain=(0.0, 0.0),
        haltere_angular_velocity_rad_s=(0.0, 0.0, 0.0),
        latency_s=0.005,
        provenance=provenance(),
        confidence=confidence(),
    )
    assert frame.latency_s == 0.005
    assert frame.availability_time_s == pytest.approx(0.006)

    with pytest.raises(SchemaValidationError, match="unit norm"):
        FeedbackFrame(
            **{**frame.__dict__, "body_orientation_quaternion_wxyz": (2.0, 0.0, 0.0, 0.0)}
        )
    with pytest.raises(SchemaValidationError, match=r"timestamp_s \+ latency_s"):
        FeedbackFrame(**{**frame.__dict__, "availability_time_s": 0.007})


def test_episode_config_result_and_coverage_are_versioned():
    perturbation = PerturbationSpec(
        target_kind="dn",
        target_id="DNp26",
        mode="silence",
        start_s=0.005,
        end_s=0.010,
    )
    config = FlightEpisodeConfig(
        episode_id="fixture",
        scenario="steady_airborne",
        duration_s=0.02,
        physics_timestep_s=0.0001,
        neural_timestep_s=0.005,
        muscle_timestep_s=0.0001,
        render_timestep_s=1.0 / 60.0,
        seed=7,
        circuit_snapshot="sha256:abc",
        model_hashes={"flybody": "sha256:def"},
        world={"gravity_m_s2": 9.81},
        perturbations=(perturbation,),
    )
    coverage = CoverageSummary(
        branch="NOD1",
        total_structural_synapses=712,
        resolved_structural_synapses=545,
        unresolved_structural_synapses=167,
        provenance=provenance(),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
        scope="fixture",
    )
    result = FlightEpisodeResult(
        config=config,
        status=EpisodeStatus.COMPLETE,
        validation_status=ValidationStatus.EXPLORATORY,
        sample_times_s=(0.0, 0.02),
        body_position_m=((0.0, 0.0, 0.01), (1e-4, 0.0, 0.01)),
        body_orientation_quaternion_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        body_linear_velocity_m_s=((0.0, 0.0, 0.0), (0.01, 0.0, 0.0)),
        body_angular_velocity_rad_s=((0.0, 0.0, 0.0), (0.0, 0.0, 0.1)),
        wing_angles_rad={
            "left": ((0.0, 0.7, 0.0), (math.pi / 2, 0.7, 0.0)),
            "right": ((0.0, 0.7, 0.0), (math.pi / 2, 0.7, 0.0)),
        },
        aerodynamic_force_n=((0.0, 0.0, 1e-5), (0.0, 0.0, 1e-5)),
        aerodynamic_moment_n_m=((0.0, 0.0, 0.0), (0.0, 0.0, 1e-9)),
        energy_j=(0.0, 1e-7),
        coverage=(coverage,),
        uncertainty={"yaw_rad_s": [0.01, 0.02]},
        numerical_diagnostics={"physics_steps": 200},
        provenance=provenance(),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
    )
    assert result.to_dict()["schema_version"] == "1.0.0"
    assert coverage.resolved_fraction == pytest.approx(545 / 712)

    with pytest.raises(SchemaValidationError, match="unit quaternions"):
        FlightEpisodeResult(
            **{
                **result.__dict__,
                "body_orientation_quaternion_wxyz": (
                    (1.0, 0.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0, 0.0),
                ),
            }
        )
    with pytest.raises(SchemaValidationError, match="endpoint"):
        FlightEpisodeResult(
            **{
                **result.__dict__,
                "sample_times_s": (0.0, 0.019),
            }
        )


def test_model_registry_rejects_duplicate_versions_and_exposes_posterior_units():
    parameter = ModelParameter(
        name="wingbeat_frequency",
        mean=200.0,
        unit="Hz",
        standard_deviation=10.0,
        lower_bound=100.0,
        upper_bound=300.0,
    )
    record = ModelRecord(
        name="virtual-wing-hinge",
        version="0.1.0",
        role="muscle state to generalized wing torque",
        artifact_hash="sha256:fixture",
        artifact_uri="urn:test:wing-hinge",
        parameters=(parameter,),
        calibration_dataset=None,
        provenance=provenance(),
        confidence=confidence(EvidenceTier.MODEL_INFERENCE),
        validation_status=ValidationStatus.EXPLORATORY,
        license="research-use fixture",
    )
    registry = ModelRegistry(
        models=(record,),
        generated_at_utc="2026-07-17T00:00:00Z",
        provenance=provenance(),
    )
    assert registry.get("virtual-wing-hinge").parameters[0].unit == "Hz"
    with pytest.raises(SchemaValidationError, match="unique"):
        ModelRegistry(
            models=(record, record),
            generated_at_utc="2026-07-17T00:00:00Z",
            provenance=provenance(),
        )


def test_checked_in_model_registry_conforms_to_public_schema():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "models"
        / "flight_model_registry.json"
    )
    registry = ModelRegistry.from_dict(json.loads(registry_path.read_text()))
    assert len(registry.models) == 3
    assert all(model.validation_status is ValidationStatus.EXPLORATORY for model in registry.models)
    assert registry.get("flygym-flybody-flight-adapter").parameters[0].unit == "s"
    repository_root = Path(__file__).resolve().parents[1]
    for model in registry.models:
        artifact = repository_root / model.artifact_uri
        assert model.artifact_hash == "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
