# Software Lock-In Amplifier (LIA)

A comprehensive software implementation of a Lock-In Amplifier with performance analysis capabilities (main branch), hardware integration capabilities, robust fallback systems, and centralized results management (hotfix/streaming branch).

### Hardware LIA reference
1.[SR860 website](https://thinksrs.com/products/sr860.html)
2.[SR860 user manual](https://www.thinksrs.com/downloads/pdfs/manuals/SR860m.pdf)

### Features (main branch)

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

(hotfix/streaming branch)

13. Software LIA Implementation: Digital demodulation, filtering, and comprehensive analysis
14. Hardware Integration: Support for real data acquisition from function generators
15. Robust Fallback System: UDP (RTB2004 do not support, remove UDP relevant codes) → TCP → Simulated data fallback
16. Centralized Results: Compare software vs hardware LIA results
17. Comprehensive Visualization: comparison dashboards
18. Real-time Streaming: Support for UDP streaming from compatible instruments

### Installation

#### Method 1: Using requirements.txt (Recommended)
```bash
# Clone repository
git clone https://github.com/your-username/software-lia.git
cd software-lia

# Install dependencies
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
# Run with default parameters (simulated data)
python softwareLIA.py

# Custom frequency and amplitude
python softwareLIA.py -f 100000 -a 0.005

# Longer acquisition with specific time constant
python softwareLIA.py -d 0.02 -tc 0.002

# all-in-one line
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

### Advanced Workflow (hotfix/streaming branch)
#### switch to streaming branch
```bash
git checkout hotfix/streaming
pip install -r requirements.txt
```

#### Enhanced Installation
```bash
# Additional dependencies for hardware integration
pip install RsInstrument python-vxi11 docopt

# For centralized results management
pip install pandas
```

#### Configure Hardware Connections
+ Ensure your instruments are properly connected:
      1. Function Generator/Oscilloscope: 192.168.0.5 (default)
      2. Hardware LIA (SR865): 172.25.98.253 (default)
      
#### Run Hardware LIA Streaming
```bash
# Basic streaming acquisition
python stream.py --file hardware_data.csv --duration 10

# Advanced streaming with specific parameters
python stream.py --address 172.25.98.253 --duration 15 --vars XY --rate 50000 --file hardware_measurement.csv
```

#### Collect and Standardize Hardware Results
```bash
# Process hardware LIA output
python hardware_results_collector.py hardware_data.csv

# This creates standardized JSON and CSV files:
# - hardware_results_YYYYMMDD_HHMMSS.json
# - hardware_results_YYYYMMDD_HHMMSS.csv

```
#### Run Software LIA with Real Data
```bash
# Basic real data acquisition
python softwareLIA.py --use-real-data

# Real data with specific instrument IP
python softwareLIA.py --use-real-data --fg-ip 192.168.0.5

# Complete analysis with hardware comparison
python softwareLIA.py --use-real-data --hardware-results hardware_results_20241201_143022.json

# Custom analysis with real data
python softwareLIA.py --use-real-data -f 100000 -a 0.01 -d 0.02 --freq-start 50000 --freq-end 150000
```

#### View Centralized Results
```bash
# Generate comprehensive comparison dashboard
python comparison_dashboard.py
```
### Fallback Behavior
+ The system automatically handles connection issues:
      1. UDP Streaming (if instrument supports it)
      2. TCP Fallback (traditional SCPI commands)
      3. Simulated Data (if hardware unavailable)
```bash
# Example: Force TCP connection only
fg = EnhancedFunctionGenerator('192.168.0.5')
if fg.connection_type == 'UDP':
    fg.close()
    # Reinitialize or use different parameters
```

### Software LIA Parameters
```bash
# Signal Parameters
-f, --frequency FLOAT        Signal frequency in Hz (default: 123000)
-a, --amplitude FLOAT        Signal amplitude in V (default: 0.01)
-d, --duration FLOAT         Signal duration in seconds (default: 0.01)
-p, --phase-offset FLOAT     Signal phase offset in degrees (default: 45)
-n, --noise-level FLOAT      Noise level (default: 0.005)

# Lock-in Parameters
-sr, --sampling-rate FLOAT   Sampling rate in Hz (default: 2e6)
-tc, --time-constant FLOAT   Lock-in time constant in seconds (default: 0.001)

# Frequency Scan
--freq-start FLOAT           Frequency scan start in Hz (default: 80000)
--freq-end FLOAT             Frequency scan end in Hz (default: 160000)
--freq-step FLOAT            Frequency scan step size in Hz (default: 500)

# Enhanced Features
--use-real-data              Use real data from function generator
--fg-ip TEXT                 Function generator IP address (default: 192.168.0.5)
--hardware-results TEXT      Path to hardware LIA results for comparison
```

### Hardware LIA Streaming Parameters
```bash
-a, --address <A>            IP address of SR865 (default: 172.25.98.253)
-d, --duration <D>           Transfer duration in seconds (default: 10)
-f, --file <F>               Output filename
-v, --vars <V>               Variables to stream: X, XY, RT, or XYRT (default: X)
-r, --rate <R>               Sample rate per second (default: 1e5)
-p, --port <P>               UDP Port (default: 1865)
```

### Output Files

```
project/
├── lia_results/                 # Centralized results directory
│   ├── software_lia_*.json     # Software LIA results
│   ├── hardware_results_*.json # Hardware LIA results  
│   ├── comparison_*.json       # Comparison reports
│   └── comparison_plot_*.png   # Comparison visualizations
├── lockin_analysis_*.png       # 8-panel analysis plots
├── lockin_analysis_*.log       # Detailed analysis logs
└── hardware_data.csv          # Raw hardware LIA data
```

### Troubleshooting

#### Hardware Connection Failed

WARNING: Function generator module not available.
Falling back to simulated data analysis.

Solution: Check instrument IP addresses and network connectivity.

#### UDP Streaming Not Working
UDP connection failed: [Error details]
Solution: The system automatically falls back to TCP. Check firewall settings for UDP port 1865.


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








