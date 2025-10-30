# Software Lock-In Amplifier (LIA)

A comprehensive software implementation of a Lock-In Amplifier with performance analysis capabilities.

### Features

- **Digital Lock-In Detection**: X, Y, R, and θ component extraction
- **Frequency Scanning**: Automatic optimal frequency detection
- **Autophase Optimization**: Automatic phase adjustment for maximum signal
- **Comprehensive Analysis**: 8-panel visualization of all performance metrics
- **Realistic Signal Simulation**: Includes noise, harmonics, and phase impairments
- **High Performance**: JIT-compiled core functions using Numba
- **Command Line Interface**: User-friendly parameter configuration
- **Automated Logging**: Detailed analysis results with timestamps
- **High-Quality Plots**: 150 DPI PNG and JPG outputs
- **Cleanup Utility**: Easy removal of generated files

### Installation

#### Method 1: Using requirements.txt (Recommended)
```bash
git clone https://github.com/xinglech/softwareLIA.git
cd softwareLIA
pip install -r requirements.txt
```
#### Method 2: Manual Installation
```bash
pip install numpy matplotlib scipy numba tqdm
```
#### Method 3: Using Conda
```bash
conda install numpy matplotlib scipy numba tqdm
```
### Verification
```bash
python softwareLIA.py --help
```

### Quick Start
#### Basic Usage
```bash
python softwareLIA.py
```

#### Custom Analysis
```bash
python softwareLIA.py -f 100000 -a 0.005 -d 0.02 --freq-start 50000 --freq-end 150000
```
### Command Line Options
Parameter	Description	Default
-f, --frequency	Signal frequency in Hz	123000
-a, --amplitude	Signal amplitude in V	0.01
-d, --duration	Signal duration in seconds	0.01
-p, --phase-offset	Phase offset in degrees	45
-n, --noise-level	Noise level	0.005
-sr, --sampling-rate	Sampling rate in Hz	2e6
-tc, --time-constant	Time constant in seconds	0.001
--freq-start	Frequency scan start in Hz	80000
--freq-end	Frequency scan end in Hz	160000
--freq-step	Frequency scan step in Hz	500
### Output Files
The program generates timestamped files:

Log file: lockin_analysis_YYYYMMDD_HHMMSS.log - Detailed analysis results

Plot files: lockin_analysis_YYYYMMDD_HHMMSS.png/jpg - 8-panel comprehensive visualization

### Example Output
text
Software Lock-In Amplifier - Starting Analysis...
==================================================
COMPREHENSIVE LOCK-IN PERFORMANCE ANALYSIS
==================================================
INPUT PARAMETERS:
  signal_frequency: 123.0 kHz
  amplitude: 10.0 mV
  duration: 10.0 ms
  sampling_rate: 2.0 MHz
  time_constant: 0.001
  noise_level: 0.005
  phase_offset: 45

FINAL RESULTS WITH UNCERTAINTIES:
Optimal Frequency (Phase): 123.000 ± 0.250 kHz
Optimal Frequency (Magnitude): 123.000 ± 0.250 kHz
X-component: 9.886 ± 0.211 mV (Error: 39.8%)
R-magnitude: 9.898 ± 0.201 mV (Error: 1.0%)
Phase: -0.6 ± 2.9° (Error: 45.6°)
SNR: -0.4 dB
Residual Noise: 9.608 mV
### Cleanup Utility
#### Remove generated output files:

### Basic cleanup (with confirmation)
```bash
python clean.py
```
#### Force delete all output files
```bash
python clean.py --force
```
#### Preview what would be deleted
```bash
python clean.py --dry-run
```
### Cleanup script options:
-f, --force: Force deletion without confirmation

-d, --dry-run: Show what would be deleted without actually deleting

-y, --yes: Auto-confirm deletion

### Troubleshooting
#### Common Issues
ModuleNotFoundError: No module named 'scipy'

```bash
pip install -r requirements.txt
```
Import errors on Windows

Ensure latest pip: python -m pip install --upgrade pip

Install Microsoft Visual C++ Build Tools if needed

### Performance issues

First run may be slower due to JIT compilation

Subsequent runs will be faster due to caching

### Theory
Lock-In Amplifier Principle
A lock-in amplifier extracts a signal at a specific reference frequency by:

Multiplying the input signal with reference sine and cosine waves

Low-pass filtering to extract DC components

Calculating magnitude (R) and phase (θ) from X and Y components

#### Mathematical Basis
text
X = (2/T) ∫ signal × cos(2πf_ref × t + φ) dt
Y = (2/T) ∫ signal × sin(2πf_ref × t + φ) dt
R = √(X² + Y²)
θ = atan2(Y, X)
Contributing
Fork the repository

Create a feature branch

Make your changes

Submit a pull request

