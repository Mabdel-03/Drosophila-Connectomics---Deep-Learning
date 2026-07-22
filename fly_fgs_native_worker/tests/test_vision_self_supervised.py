import hashlib
import math

import numpy as np
import pytest

from fly_sensor2behavior.vision import (
    AnalyticGratingScene,
    FigureGroundScene,
    PanoramicRetina,
    RandomColumnScene,
    ReichardtMotionDetector,
    UniformScene,
)


def _motion_response(scene, *, frames=40):
    retina = PanoramicRetina(
        receptor_count=64,
        sensor_latency_s=0.0,
        integration_subsamples=3,
    )
    detector = ReichardtMotionDetector(output_latency_s=0.0)
    responses = []
    for index in range(frames):
        frame = retina.sample(
            scene,
            exposure_start_s=index * 0.002,
            exposure_end_s=(index + 1) * 0.002,
        )
        responses.append(
            detector.update(frame, current_time_s=frame.availability_time_s).population_response
        )
    return np.asarray(responses)


def test_uniform_luminance_has_no_sustained_motion_drive():
    response = _motion_response(UniformScene(0.5))
    assert np.max(np.abs(response)) < 1.0e-15


def test_full_field_flicker_has_no_direction_preference():
    retina = PanoramicRetina(receptor_count=32, sensor_latency_s=0.0)
    detector = ReichardtMotionDetector(output_latency_s=0.0)
    values = []
    for index in range(20):
        frame = retina.sample(
            UniformScene(0.2 if index % 2 else 0.8),
            exposure_start_s=index * 0.002,
            exposure_end_s=(index + 1) * 0.002,
        )
        values.append(detector.update(frame).population_response)
    assert np.max(np.abs(values)) < 1.0e-15


def test_preferred_and_null_grating_have_opposite_equal_response():
    positive = np.mean(
        _motion_response(AnalyticGratingScene(angular_velocity_rad_s=1.0))[10:]
    )
    negative = np.mean(
        _motion_response(AnalyticGratingScene(angular_velocity_rad_s=-1.0))[10:]
    )
    zero = np.mean(
        _motion_response(AnalyticGratingScene(angular_velocity_rad_s=0.0))[10:]
    )
    assert positive > 0.0
    assert negative < 0.0
    assert positive == pytest.approx(-negative, rel=1.0e-12, abs=1.0e-12)
    assert zero == pytest.approx(0.0, abs=1.0e-15)


def test_speed_sweep_is_continuous_and_signed_through_zero():
    speeds = (-0.2, -0.1, 0.0, 0.1, 0.2)
    responses = np.asarray(
        [
            np.mean(
                _motion_response(
                    AnalyticGratingScene(angular_velocity_rad_s=speed)
                )[10:]
            )
            for speed in speeds
        ]
    )
    assert np.all(np.diff(responses) > 0.0)
    assert responses[2] == pytest.approx(0.0, abs=1.0e-15)
    assert responses[1] == pytest.approx(-responses[3], rel=1.0e-12)


def test_phase_zero_and_two_pi_are_exactly_equivalent():
    retina = PanoramicRetina(receptor_count=64, sensor_latency_s=0.0)
    first = retina.sample(
        AnalyticGratingScene(phase_rad=0.0, angular_velocity_rad_s=0.7),
        exposure_start_s=0.010,
        exposure_end_s=0.012,
    )
    second = retina.sample(
        AnalyticGratingScene(phase_rad=2.0 * math.pi, angular_velocity_rad_s=0.7),
        exposure_start_s=0.010,
        exposure_end_s=0.012,
    )
    assert np.array_equal(first.samples, second.samples)


def test_figure_center_and_texture_phase_are_continuous_at_delayed_onset():
    scene = FigureGroundScene(
        background_velocity_rad_s=0.3,
        figure_velocity_rad_s=-0.8,
        figure_onset_s=0.25,
    )
    epsilon = 1.0e-9
    before = scene.figure_center_rad(scene.figure_onset_s - epsilon)
    at = scene.figure_center_rad(scene.figure_onset_s)
    after = scene.figure_center_rad(scene.figure_onset_s + epsilon)
    assert before == pytest.approx(at, abs=1.0e-8)
    assert after == pytest.approx(at, abs=1.0e-8)
    azimuth = np.linspace(-math.pi, math.pi, 257, endpoint=False)
    just_before = scene.luminance(azimuth, scene.figure_onset_s - epsilon)
    at_onset = scene.luminance(azimuth, scene.figure_onset_s)
    assert np.max(np.abs(just_before - at_onset)) < 1.0e-7


def test_random_columns_are_seed_reproducible_and_not_mislabeled_alternation():
    first = RandomColumnScene(seed=7, column_count=64).texture
    second = RandomColumnScene(seed=7, column_count=64).texture
    third = RandomColumnScene(seed=8, column_count=64).texture
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    alternating = np.resize(np.asarray([0.1, 0.9]), first.shape)
    assert not np.array_equal(first, alternating)
    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(
        second.tobytes()
    ).hexdigest()


def test_motion_detector_rejects_future_and_out_of_order_frames():
    retina = PanoramicRetina(receptor_count=16, sensor_latency_s=0.005)
    first = retina.sample(
        UniformScene(), exposure_start_s=0.0, exposure_end_s=0.001
    )
    detector = ReichardtMotionDetector()
    with pytest.raises(ValueError, match="not yet causally available"):
        detector.update(first, current_time_s=0.001)
    detector.update(first, current_time_s=first.availability_time_s)
    with pytest.raises(ValueError, match="increasing measurement"):
        detector.update(first, current_time_s=first.availability_time_s)
