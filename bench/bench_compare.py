"""
Benchmarking scripts comparing pure-Python, numpy, and streaming moving-average implementations.
"""
import time
from statistics import mean

from dsp_portfolio_improved import (
    generate_signal,
    inject_gaussian_noise,
    apply_causal_lpf_python,
    apply_causal_lpf_numpy,
    apply_causal_lpf_streaming,
)


def time_function(fn, *args, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return mean(times), times


def main():
    TARGET_FREQ = 5.0
    FS = 100.0
    DURATION = 10.0  # longer signal to amplify perf differences
    NOISE_LEVEL = 0.35
    RNG_SEED = 123
    WINDOW = 5

    clean = generate_signal(TARGET_FREQ, FS, DURATION)
    noisy = inject_gaussian_noise(clean, NOISE_LEVEL, seed=RNG_SEED)

    # Pure Python
    py_mean, py_times = time_function(apply_causal_lpf_python, noisy, WINDOW)

    # Numpy (if available)
    np_mean, np_times = time_function(apply_causal_lpf_numpy, noisy, WINDOW)

    # Streaming (generator)
    def run_stream():
        list(apply_causal_lpf_streaming(noisy, WINDOW))

    st_mean, st_times = time_function(run_stream)

    print("Benchmark results (mean over runs):")
    print(f"Pure Python     : {py_mean:.6f} s (runs: {py_times})")
    print(f"Numpy/vectorized: {np_mean:.6f} s (runs: {np_times})")
    print(f"Streaming (deque): {st_mean:.6f} s (runs: {st_times})")

    try:
        speedup_np = py_mean / np_mean
        speedup_stream = py_mean / st_mean
        print(f"Speedup (python -> numpy): {speedup_np:.2f}x")
        print(f"Speedup (python -> streaming): {speedup_stream:.2f}x")
    except Exception:
        pass


if __name__ == "__main__":
    main()
