import numpy as np

from fly_sensor2behavior.paper_signal import (
    design_paper_antialias_fir,
    measure_fir_response,
    synchronize_raw_torque,
)


def test_released_fir_meets_passband_and_stopband_contract():
    for rate in (40000, 80000):
        taps, specification = design_paper_antialias_fir(rate)
        response = measure_fir_response(taps, rate)
        assert len(taps) % 2 == 1
        assert response["passband_ripple_db"] <= 0.01
        assert response["stopband_attenuation_db"] >= 100.0
        assert specification.group_delay_input_samples == (len(taps) - 1) // 2


def test_synchronizer_preserves_2_5_and_200_hz_and_rejects_above_nyquist():
    rate = 40000
    time = np.arange(-0.4 + 1 / rate, 0.8 + 0.4 + 1 / rate, 1 / rate)
    signal = (
        np.sin(2 * np.pi * 2.5 * time)
        + 0.25 * np.sin(2 * np.pi * 200 * time)
        + 0.5 * np.sin(2 * np.pi * 3000 * time)
    )
    synchronized_time, synchronized, receipt = synchronize_raw_torque(
        time,
        signal,
        trial_duration_s=0.8,
        input_rate_hz=rate,
    )
    expected = (
        np.sin(2 * np.pi * 2.5 * synchronized_time)
        + 0.25 * np.sin(2 * np.pi * 200 * synchronized_time)
    )
    assert np.sqrt(np.mean((synchronized - expected) ** 2)) < 1.0e-3
    assert receipt["group_delay_compensated"] is True
