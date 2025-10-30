# Software Lock-In Amplifier (LIA)

A comprehensive software implementation of a Lock-In Amplifier with performance analysis capabilities.

## Features

- **Digital Lock-In Detection**: X, Y, R, and θ component extraction
- **Frequency Scanning**: Automatic optimal frequency detection
- **Autophase Optimization**: Automatic phase adjustment for maximum signal
- **Comprehensive Analysis**: 8-panel visualization of all performance metrics
- **Realistic Signal Simulation**: Includes noise, harmonics, and phase impairments
- **High Performance**: JIT-compiled core functions using Numba

## Installation

### Using requirements.txt (Recommended)
1. Clone the repository:
```bash
git clone https://github.com/yourusername/software-LIA.git
cd software-LIA

## Cleanup Utility

A separate cleanup script is provided to remove generated output files:

### Basic cleanup (with confirmation)
```bash

python clean.py
