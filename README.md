# Software Lock-In Amplifier (LIA)

A comprehensive software implementation of a Lock-In Amplifier with performance analysis capabilities.

### Hardware LIA reference
1.[SR860 website](https://thinksrs.com/products/sr860.html)
2.[SR860 user manual](https://www.thinksrs.com/downloads/pdfs/manuals/SR860m.pdf)

### Features

1. **Digital Lock-In Detection**: X, Y, R, and θ component extraction
2. **Frequency Scanning**: Automatic optimal frequency detection
3. **Autophase Optimization**: Automatic phase adjustment for maximum signal
4. **Comprehensive Analysis**: 8-panel visualization of all performance metrics
      ![fig_scenarios.pdf](https://github.com/user-attachments/files/23264236/fig_scenarios.pdf)
6. **Realistic Signal Simulation**: Includes noise, harmonics, and phase impairments
7. **Fast Execution**: JIT-compiled core functions using Numba
      [Compiling Python code with @ji](https://numba.readthedocs.io/en/stable/user/jit.html)
      [Numba website](https://numba.pydata.org/)
9. **Command Line Interface**: User-friendly parameter configuration
10. **Automated Logging**: Detailed analysis results with timestamps
11. **High-Quality Plots**: 150 DPI PNG and JPG outputs
12. **Cleanup Utility**: Easy removal of generated files

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

#### Basic Usage (run scanning with default parameters)
```bash
python softwareLIA.py
```

#### Custom Analysis
```bash
python softwareLIA.py -f 100000 -a 0.005 -d 0.02 --freq-start 50000 --freq-end 150000
```
### Command Line Options
Parameter	Description	**Default**
- -f, --frequency	Signal frequency in **Hz**	123000
- -a, --amplitude	Signal amplitude in **V**	0.01
- -d, --duration	Signal duration in **seconds**	0.01
- -p, --phase-offset	Phase offset in **degrees**	45
- -n, --noise-level	Noise level	0.005
- -sr, --sampling-rate	Sampling rate in **Hz**	2e6
- -tc, --time-constant	Time constant in **seconds**	0.001
- --freq-start	Frequency scan start in **Hz**	80000
- --freq-end	Frequency scan end in **Hz**	160000
- --freq-step	Frequency scan step in **Hz**	500

### Output Files
The program generates **timestamped** files:

1. **Log file**: lockin_analysis_YYYYMMDD_HHMMSS.log - Detailed analysis results
2. **Plot files**: lockin_analysis_YYYYMMDD_HHMMSS.png/jpg - 8-panel comprehensive visualization

### Example Output

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
#### Force delete all output files (do not recommend)
```bash
python clean.py --force
```
#### **Preview** what would be deleted (Recommended)
```bash
python clean.py --dry-run
```
### Cleanup script options:
+ -f, --force: Force deletion without confirmation
+ -d, --dry-run: Show what would be deleted without actually deleting
+ -y, --yes: Auto-confirm deletion

### Troubleshooting

#### Common Issues
+ ModuleNotFoundError: No module named 'scipy'
```bash
pip install -r requirements.txt
```
+ Import errors on Windows:
    1.Ensure latest pip: python -m pip install --upgrade pip
    2.Install Microsoft Visual C++ Build Tools if needed

### Performance issues

+ First run may be slower due to *JIT compilation*
+ Subsequent runs will be faster due to *caching*

### Theory

    Lock-In Amplifier Principle
    ===========================
    A lock-in amplifier extracts a signal at a specific reference frequency by:
    
    1.Multiplying the input signal with reference sine and cosine waves
    2.Low-pass filtering to extract DC components
    3.Calculating magnitude (R) and phase (θ) from X and Y components

#### Mathematical Basis

$$
X = \frac{2}{T} \int \text{signal} \times \cos(2\pi f_{\text{ref}} \times t + \phi)  dt
$$

$$
Y = \frac{2}{T} \int \text{signal} \times \sin(2\pi f_{\text{ref}} \times t + \phi)  dt
$$

$$
R = \sqrt{X^2 + Y^2}
$$

$$
\theta = \text{atan2}(Y, X)
$$

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request





