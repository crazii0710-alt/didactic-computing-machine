import math
import time
from dsp_portfolio_improved import (
    generate_signal,
    inject_gaussian_noise,
    apply_causal_lpf_python,
    apply_causal_lpf_numpy,
    apply_causal_lpf_streaming,
    calculate_snr,
)


def test_generate_signal_length():
    sig = generate_signal(5, 100, 1)
    assert len(sig) == 100


def test_inject_noise_reproducible():
    clean = generate_signal(5, 100, 0.5)
    a = inject_gaussian_noise(clean, noise_level=0.1, seed=123)
    b = inject_gaussian_noise(clean, noise_level=0.1, seed=123)
    assert a == b


def test_moving_average_matches_numpy():
    clean = generate_signal(5, 100, 0.5)
    noisy = inject_gaussian_noise(clean, noise_level=0.2, seed=7)
    py = apply_causal_lpf_python(noisy, window_size=5)
    npy = apply_causal_lpf_numpy(noisy, window_size=5)
    # numeric tolerance
    assert len(py) == len(npy)
    for x, y in zip(py, npy):
        assert abs(x - y) < 1e-9


def test_streaming_equivalence():
    clean = generate_signal(5, 100, 0.5)
    noisy = inject_gaussian_noise(clean, noise_level=0.2, seed=9)
    streamed = list(apply_causal_lpf_streaming(noisy, window_size=5))
    py = apply_causal_lpf_python(noisy, window_size=5)
    assert len(streamed) == len(py)
    for x, y in zip(streamed, py):
        assert abs(x - y) < 1e-9


def test_snr_monotonic_improvement():
    clean = generate_signal(5, 100, 1.0)
    noisy = inject_gaussian_noise(clean, noise_level=0.5, seed=42)
    filtered = apply_causal_lpf_numpy(noisy, window_size=5)
    snr_noisy = calculate_snr(clean, noisy)
    snr_filtered = calculate_snr(clean, filtered)
    # Filtering should not make SNR drastically worse in this setup
    assert snr_filtered >= snr_noisy - 1e-6


if __name__ == "__main__":
    start = time.time()
    test_generate_signal_length()
    test_inject_noise_reproducible()
    test_moving_average_matches_numpy()
    test_streaming_equivalence()
    test_snr_monotonic_improvement()
    print("All tests passed. Time:", time.time() - start)
