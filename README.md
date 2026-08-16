# Didactic Computing Machine — DSP Portfolio

This repository contains a small project showcasing real-time causal digital signal processing techniques and performance analytics. It includes:

- dsp_portfolio_improved.py: Improved DSP utilities (generate signal, AWGN, causal moving-average filter) with numpy-optimized and streaming implementations.
- firwin_example.py: Example showing FIR design using scipy.signal.firwin and SNR comparison.
- tests/: Pytest unit tests for the DSP utilities.
- bench/bench_compare.py: Simple benchmark comparing pure-Python, numpy, and streaming filter implementations.
- notebook/DSP_portfolio_analysis.ipynb: Jupyter notebook with examples and plots.

Requirements
------------
Install the development requirements:

pip install -r requirements.txt

Usage
-----
Run the main improved script:

python dsp_portfolio_improved.py

Run the firwin example (requires scipy):

python firwin_example.py

Run tests:

pytest -q

Run benchmark:

python bench/bench_compare.py

Notes
-----
- The moving-average filter implemented here is causal and uses explicit zero-padding for t < 0. If you prefer edge-aware averaging that divides by the number of available samples at startup, modify the implementations accordingly.
- For better filters (sharper cutoff, controlled ripple), see firwin_example.py which uses scipy.signal.firwin.
