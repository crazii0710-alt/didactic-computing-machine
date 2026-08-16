"""
FIR design example using scipy.signal.firwin and frequency response analysis.
Demonstrates designing a windowed-sinc FIR lowpass, applying it to a noisy signal,
and computing SNR improvement versus the simple moving-average.
"""
from typing import List
import math

try:
    import numpy as np
    from scipy import signal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from dsp_portfolio_improved import (
    generate_signal,
    inject_gaussian_noise,
    calculate_snr,
)


def firwin_lpf_design(cutoff_hz: float, fs: float, numtaps: int = 51, window: str = "hamming") -> List[float]:
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for firwin example")
    nyq = fs / 2.0
    taps = signal.firwin(numtaps, cutoff_hz / nyq, window=window)
    return taps.tolist()


def apply_fir_filter(signal_in: List[float], taps: List[float]) -> List[float]:
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for firwin example")
    import numpy as _np
    x = _np.asarray(signal_in)
    y = _np.convolve(x, _np.asarray(taps), mode="same")
    return y.tolist()


if __name__ == "__main__":
    TARGET_FREQ = 5.0
    FS = 100.0
    DURATION = 2.0
    NOISE_LEVEL = 0.35
    RNG_SEED = 42

    clean = generate_signal(TARGET_FREQ, FS, DURATION)
    noisy = inject_gaussian_noise(clean, NOISE_LEVEL, seed=RNG_SEED)

    if SCIPY_AVAILABLE:
        taps = firwin_lpf_design(cutoff_hz=12.0, fs=FS, numtaps=63)
        filtered = apply_fir_filter(noisy, taps)

        snr_noisy = calculate_snr(clean, noisy)
        snr_ma = calculate_snr(clean, filtered)  # using fir as comparator

        print(f"SNR noisy: {snr_noisy:.4f} dB")
        print(f"SNR FIR (firwin): {snr_ma:.4f} dB")
    else:
        print("scipy not available; install scipy to run firwin_example.py")
