"""
Math + ECE Portfolio Project: Real-Time Causal Digital Signal Processing & Performance Analytics
Author: [Your Name] (improved version)
Notes:
 - Vectorized numpy implementation for batch processing
 - Streaming implementation (deque) for real-time / online processing
 - Reproducible RNG via seed
 - Robust SNR calculation with epsilon
"""
from collections import deque
from typing import List, Iterable, Generator, Optional
import math
import random

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def generate_signal(frequency: float, sample_rate: float, duration: float) -> List[float]:
    """Generate a sampled sine wave."""
    time_steps = int(sample_rate * duration)
    return [math.sin(2 * math.pi * frequency * (t / sample_rate)) for t in range(time_steps)]


def inject_gaussian_noise(signal: List[float], noise_level: float = 0.3, seed: Optional[int] = None) -> List[float]:
    """Add AWGN to a signal. Use seed for reproducibility."""
    rng = random.Random(seed)
    return [x + rng.gauss(0, noise_level) for x in signal]


def apply_causal_lpf_python(noisy: List[float], window_size: int = 5) -> List[float]:
    """
    Causal moving-average (FIR) implemented in pure Python.
    Explicit zero-padding is applied for t < 0, and we divide by window_size.
    This is O(N * window_size) and fine for small signals or educational use.
    """
    filtered = []
    N = window_size
    for i in range(len(noisy)):
        acc = 0.0
        # sum x[i], x[i-1], ..., x[i-N+1] (zeros for negative indices)
        for k in range(N):
            idx = i - k
            if idx >= 0:
                acc += noisy[idx]
            else:
                acc += 0.0
        filtered.append(acc / N)
    return filtered


def apply_causal_lpf_numpy(noisy: List[float], window_size: int = 5) -> List[float]:
    """
    Vectorized causal moving average using numpy:
    y = convolve(x, h) where h = ones(N)/N, and we take the first len(x) samples
    so that the filter is causal with implicit zero-padding for t<0.
    """
    if not NUMPY_AVAILABLE:
        return apply_causal_lpf_python(noisy, window_size)
    x = np.asarray(noisy, dtype=float)
    h = np.ones(window_size, dtype=float) / window_size
    y_full = np.convolve(x, h, mode="full")
    y = y_full[: len(x)]
    return y.tolist()


def apply_causal_lpf_streaming(stream: Iterable[float], window_size: int = 5) -> Generator[float, None, None]:
    """
    Streaming (real-time) causal moving-average using a deque.
    Yields filtered output sample-by-sample. Good for long or unbounded streams.
    """
    buf = deque(maxlen=window_size)
    acc = 0.0
    N = window_size
    for x in stream:
        if len(buf) == N:
            # buffer full; remove oldest
            oldest = buf[0]
            acc -= oldest
        buf.append(x)
        acc += x
        # For causal zero-padded behavior we always divide by N (window_size)
        yield acc / N


def calculate_snr(clean: List[float], evaluated: List[float], eps: float = 1e-12) -> float:
    """
    Compute SNR (dB) = 10 * log10(P_signal / P_noise).
    Uses an eps to avoid division by zero.
    """
    if len(clean) != len(evaluated):
        raise ValueError("clean and evaluated must have same length")
    signal_power = sum(x * x for x in clean) / len(clean)
    residual = [evaluated[i] - clean[i] for i in range(len(clean))]
    noise_power = sum(x * x for x in residual) / len(residual)
    if noise_power <= eps:
        return float("inf")
    snr_linear = signal_power / noise_power
    return 10.0 * math.log10(snr_linear)


if __name__ == "__main__":
    # System constraints / parameters
    TARGET_FREQ = 5.0
    SAMPLING_RATE = 100.0
    DURATION_SEC = 2.0
    WINDOW_SIZE = 5
    NOISE_LEVEL = 0.35
    RNG_SEED = 42

    clean = generate_signal(TARGET_FREQ, SAMPLING_RATE, DURATION_SEC)
    noisy = inject_gaussian_noise(clean, NOISE_LEVEL, seed=RNG_SEED)

    # Choose processing method: numpy if available else python
    filtered = apply_causal_lpf_numpy(noisy, WINDOW_SIZE)

    snr_initial = calculate_snr(clean, noisy)
    snr_final = calculate_snr(clean, filtered)
    snr_gain = snr_final - snr_initial

    print("=========================================================")
    print("   MATH & ECE SIGNAL ANALYSIS ARCHITECTURE (IMPROVED)     ")
    print("=========================================================\n")
    print(f"[-] Samples                : {len(clean)}")
    print(f"[-] Sampling rate (Hz)     : {SAMPLING_RATE}")
    print(f"[-] Target freq (Hz)       : {TARGET_FREQ}")
    print(f"[-] Filter window (N)      : {WINDOW_SIZE}")
    print(f"[-] Nulls at k*fs/N (Hz)   : {[k * SAMPLING_RATE / WINDOW_SIZE for k in range(1,3)]}")
    print(f"[-] RNG seed               : {RNG_SEED}\n")
    print(f"[+] Initial Channel SNR    : {snr_initial:.4f} dB")
    print(f"[+] Post-Filtered SNR      : {snr_final:.4f} dB")
    print(f"[#] NET QUANTIFIABLE GAIN  : {snr_gain:.4f} dB")
    print("\n=========================================================")

    # Optional quick plot if numpy & matplotlib available
    try:
        import matplotlib.pyplot as plt  # type: ignore
        t = [i / SAMPLING_RATE for i in range(len(clean))]
        plt.figure(figsize=(10, 6))
        plt.plot(t, clean, label="clean", linewidth=1)
        plt.plot(t, noisy, label="noisy", alpha=0.6)
        plt.plot(t, filtered, label="filtered (causal MA)", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.legend()
        plt.title("Signal, Noisy, and Post-Filtered Waves")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass
