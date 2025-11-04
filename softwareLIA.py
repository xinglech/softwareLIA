#!/usr/bin/env python3
"""
Software Lock-In Amplifier (LIA) - Comprehensive Performance Analysis
Author: NSLX team
Version: 3.0 (1.0, 2.0 not shown here)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi
import time as time_module
from tqdm import tqdm
from numba import jit
import logging
import argparse
import sys
import os
from datetime import datetime

# Configure logging to both console and file
def setup_logging(log_file='lockin_analysis.log'):
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# JIT-compiled core functions
@jit(nopython=True, cache=True)
def demodulation_core_jit(signal, time, f_ref, harmonic_n, phase_offset):
    phase = 2 * np.pi * f_ref * harmonic_n * time + phase_offset
    ref_cos = np.cos(phase)
    ref_sin = np.sin(phase)
    
    X_unfiltered = signal * ref_cos * 2
    Y_unfiltered = signal * ref_sin * 2
    
    return X_unfiltered, Y_unfiltered

@jit(nopython=True, cache=True)
def calculate_amplitude_phase_jit(X, Y):
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return R, theta

class OptimizedLockInAmplifier:
    def __init__(self, fs=2e6, time_constant=0.001, filter_order=4):
        self.fs = fs
        self.time_constant = time_constant
        self.cutoff_freq = 1.0 / (2 * np.pi * time_constant)
        self.filter_order = filter_order
        self.phase_offset = 0.0
        self.harmonic_n = 1
        
        self._initialize_sos_filter()
        self.reset_filter()
        
        logging.info(f"Lock-In: fs={fs/1e6:.1f}MHz, τ={time_constant*1000:.1f}ms, fc={self.cutoff_freq:.1f}Hz")

    def _initialize_sos_filter(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff_freq / nyq
        normal_cutoff = np.clip(normal_cutoff, 1e-6, 0.2)
        
        self.sos = butter(self.filter_order, normal_cutoff, btype='low', 
                         analog=False, output='sos')
        self.initial_zi = sosfilt_zi(self.sos)

    def reset_filter(self):
        self.x_filter_state = self.initial_zi.copy()
        self.y_filter_state = self.initial_zi.copy()

    def demodulation_core(self, signal, time, f_ref):
        return demodulation_core_jit(signal, time, f_ref, self.harmonic_n, self.phase_offset)

    def lowpass_filter(self, data, filter_state):
        rms = np.sqrt(np.mean(data**2))
        scaling = 1.0 / rms if rms > 1e-3 else 1.0
        filtered, new_state = sosfilt(self.sos, data * scaling, zi=filter_state)
        return filtered / scaling, new_state

    def demodulate(self, signal, time, f_ref):
        X_unfiltered, Y_unfiltered = self.demodulation_core(signal, time, f_ref)
        X, self.x_filter_state = self.lowpass_filter(X_unfiltered, self.x_filter_state)
        Y, self.y_filter_state = self.lowpass_filter(Y_unfiltered, self.y_filter_state)
        R, theta = calculate_amplitude_phase_jit(X, Y)
        return X, Y, R, theta

    def autophase(self, signal, time, f_ref, phase_steps=90):
        best_phase = 0.0
        max_signal = -np.inf
        original_phase = self.phase_offset
        original_state = self.x_filter_state.copy()
        
        phases = np.linspace(0, 2*np.pi, phase_steps)
        for phase in tqdm(phases, desc="Autophase"):
            self.phase_offset = phase
            self.x_filter_state = original_state.copy()
            X, _, _, _ = self.demodulate(signal, time, f_ref)
            x_steady = np.mean(X[-500:])
            if x_steady > max_signal:
                max_signal = x_steady
                best_phase = phase
        
        self.phase_offset = best_phase
        self.reset_filter()
        return best_phase

    def scan_reference_frequency(self, signal, time, freq_range, step_size):
        start_freq, end_freq = freq_range
        frequencies = np.arange(start_freq, end_freq, step_size)
        phase_responses = []
        magnitude_responses = []
        execution_times = []
        residual_noises = []
        
        original_phase = self.phase_offset
        
        for f_ref in tqdm(frequencies, desc="Frequency Scan"):
            self.reset_filter()
            start_time = time_module.perf_counter()
            X, Y, R, theta = self.demodulate(signal, time, f_ref)
            end_time = time_module.perf_counter()
            
            steady_start = int(len(X) * 0.3)
            phase_response = np.abs(np.mean(X[steady_start:]))
            magnitude_response = np.sqrt(np.mean(R[steady_start:]**2))
            
            # Calculate residual noise
            reconstructed = R * np.cos(2 * np.pi * f_ref * time + theta)
            residual_noise = np.std(signal - reconstructed)
            
            phase_responses.append(phase_response)
            magnitude_responses.append(magnitude_response)
            execution_times.append(end_time - start_time)
            residual_noises.append(residual_noise)
        
        self.phase_offset = original_phase
        self.reset_filter()
        
        # Find optimal frequencies for both methods
        opt_freq_phase = frequencies[np.argmax(phase_responses)]
        opt_freq_mag = frequencies[np.argmax(magnitude_responses)]
        
        return (opt_freq_phase, opt_freq_mag, frequencies, phase_responses, 
                magnitude_responses, execution_times, residual_noises)

def generate_realistic_signal(f_signal=123000, amplitude=0.01, duration=0.01, noise_level=0.005, phase_offset=45):
    """Generate test signal with realistic impairments"""
    fs = 2e6
    
    t = np.linspace(0, duration, int(fs * duration))
    
    # Clean signal with phase offset
    clean_signal = amplitude * np.sin(2 * np.pi * f_signal * t + np.radians(phase_offset))
    
    # Realistic impairments
    noise = noise_level * np.random.randn(len(t))
    harmonic_distortion = 0.001 * np.sin(2 * np.pi * 2 * f_signal * t)
    phase_noise = 0.001 * np.random.randn(len(t))
    dc_offset = 0.002
    
    impaired_signal = (clean_signal + noise + harmonic_distortion + 
                      dc_offset + 0.0005 * phase_noise * np.cos(2 * np.pi * f_signal * t))
    
    return impaired_signal, t, f_signal, clean_signal

def create_comprehensive_plots(signal, time, clean, X, Y, R, theta, reconstructed, residual,
                              frequencies, phase_responses, magnitude_responses, 
                              execution_times, residual_noises, opt_freq_phase, opt_freq_mag,
                              input_params):
    """Create 8-panel comprehensive analysis plots"""
    
    # Create the 8-panel figure structure
    fig = plt.figure(figsize=(20, 15))
    
    # Calculate metrics for plots
    freqs_khz = frequencies / 1000
    time_ms = time * 1000
    steady_start = int(len(X) * 0.7)
    
    # Final steady-state values with uncertainties
    x_steady = np.mean(X[steady_start:])
    y_steady = np.mean(Y[steady_start:])
    r_steady = np.mean(R[steady_start:])
    theta_steady = np.mean(theta[steady_start:])
    
    x_std = np.std(X[steady_start:])
    y_std = np.std(Y[steady_start:])
    r_std = np.std(R[steady_start:])
    theta_std = np.std(theta[steady_start:])
    
    # Calculate SNR across frequency range
    snr_values = 20 * np.log10(np.array(magnitude_responses) / (np.array(residual_noises) + 1e-9))

    # =======================================================================
    # PANEL 1: Frequency Scan - Phase vs Magnitude
    # =======================================================================
    ax1 = plt.subplot(3, 3, 1)
    
    # Dual Y-axis plot
    ax1a = ax1
    ax1b = ax1.twinx()
    
    # Phase response (left axis)
    line1 = ax1a.plot(freqs_khz, np.array(phase_responses)*1000, 'b-', linewidth=2, 
                     label='Phase (X)')
    ax1a.set_xlabel('Frequency (kHz)')
    ax1a.set_ylabel('Phase Response (mV)', color='b')
    ax1a.tick_params(axis='y', labelcolor='b')
    
    # Magnitude response (right axis)  
    line2 = ax1b.plot(freqs_khz, np.array(magnitude_responses)*1000, 'r-', linewidth=2,
                     label='Magnitude (R)')
    ax1b.set_ylabel('Magnitude Response (mV)', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')
    
    # Optimal frequency markers - create separate legend entries
    line3 = ax1a.axvline(x=opt_freq_phase/1000, color='blue', linestyle='--', linewidth=2,
                 label=f'X Optimal: {opt_freq_phase/1000:.3f} kHz')
    line4 = ax1a.axvline(x=opt_freq_mag/1000, color='red', linestyle='--', linewidth=2,
                 label=f'R Optimal: {opt_freq_mag/1000:.3f} kHz')
    line5 = ax1a.axvline(x=input_params['signal_frequency']/1000, color='green', linestyle=':', linewidth=2,
                 label=f'Actual: {input_params["signal_frequency"]/1000:.1f} kHz')
    
    # Combine all legend entries
    lines = line1 + line2 + [line3, line4, line5]
    labels = [l.get_label() for l in lines]
    ax1a.legend(lines, labels, loc='upper right', fontsize=8)
    
    ax1a.set_title('1. Frequency Scan: Phase vs Magnitude', fontweight='bold', fontsize=12)
    ax1a.grid(True, alpha=0.3)
    
    # =======================================================================
    # PANEL 2: Lissajous Plot
    # =======================================================================
    ax2 = plt.subplot(3, 3, 2)
    
    # Use steady-state data only
    x_steady_data = X[steady_start:]
    y_steady_data = Y[steady_start:]
    
    # Normalize for better visualization
    x_norm = x_steady_data / np.max(np.abs(x_steady_data))
    y_norm = y_steady_data / np.max(np.abs(y_steady_data))
    
    # Scatter plot of X vs Y
    ax2.plot(x_norm[::10]*1000, y_norm[::10]*1000, 'b.', alpha=0.3, markersize=2)
    
    # Mean point
    ax2.plot(np.mean(x_norm)*1000, np.mean(y_norm)*1000, 'ro', markersize=8, 
             label='Steady State Mean')
    
    # Reference circle
    circle_theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(circle_theta)*1000, np.sin(circle_theta)*1000, 'k--', alpha=0.5, 
             linewidth=1, label='Ideal Circle')
    
    # Error ellipse (2σ)
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((np.mean(x_norm)*1000, np.mean(y_norm)*1000), 
                     width=2*np.std(x_norm)*1000, height=2*np.std(y_norm)*1000,
                     alpha=0.3, color='red', label='2σ Uncertainty')
    ax2.add_patch(ellipse)
    
    ax2.set_xlabel('X Component (mV, normalized)')
    ax2.set_ylabel('Y Component (mV, normalized)')
    ax2.set_title('2. Lissajous Plot', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # =======================================================================
    # PANEL 3: XYR and Phase Components with Error Bars
    # =======================================================================
    ax3 = plt.subplot(3, 3, 3)
    
    components = ['X', 'Y', 'R', 'Phase']
    values = [x_steady*1000, y_steady*1000, r_steady*1000, np.degrees(theta_steady)]
    errors = [x_std*1000, y_std*1000, r_std*1000, np.degrees(theta_std)]
    colors = ['blue', 'red', 'green', 'purple']
    
    bars = ax3.bar(components, values, yerr=errors, capsize=5, alpha=0.7, 
                   color=colors, edgecolor='black', linewidth=1)
    
    # Add target lines and annotations
    true_amplitude = input_params['amplitude'] * 1000
    ax3.axhline(y=true_amplitude, color='green', linestyle='--', alpha=0.7, 
                label=f'Target: {true_amplitude:.1f} mV')
    ax3.axhline(y=true_amplitude/np.sqrt(2), color='blue', linestyle='--', alpha=0.7,
                label=f'Expected X: {true_amplitude/np.sqrt(2):.1f} mV')
    ax3.axhline(y=input_params['phase_offset'], color='purple', linestyle='--', alpha=0.7,
                label=f'Target Phase: {input_params["phase_offset"]}°')
    
    # Add value labels on bars
    for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + error + 0.5,
                f'{value:.2f}±{error:.2f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    
    ax3.set_ylabel('Amplitude (mV) / Phase (°)')
    ax3.set_title('3. XYR and Phase Components', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # =======================================================================
    # PANEL 4: Signal Reconstruction
    # =======================================================================
    ax4 = plt.subplot(3, 3, 4)
    
    # Show detailed reconstruction around steady state
    display_start = int(len(time) * 0.6)
    window_size = min(1000, len(time) - display_start)
    time_window = time_ms[display_start:display_start+window_size]
    
    ax4.plot(time_window, clean[display_start:display_start+window_size]*1000, 
             'g-', linewidth=2, label='Clean Signal', alpha=0.8)
    ax4.plot(time_window, signal[display_start:display_start+window_size]*1000, 
             'k-', linewidth=1, label='Noisy Input', alpha=0.5)
    ax4.plot(time_window, reconstructed[display_start:display_start+window_size]*1000, 
             'r--', linewidth=1.5, label='Reconstructed', alpha=0.9)
    
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Amplitude (mV)')
    ax4.set_title('4. Signal Reconstruction', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # =======================================================================
    # PANEL 5: SNR vs Frequency
    # =======================================================================
    ax5 = plt.subplot(3, 3, 5)
    
    ax5.plot(freqs_khz, snr_values, 'co-', markersize=3, alpha=0.8, linewidth=1.5)
    ax5.axvline(x=opt_freq_phase/1000, color='blue', linestyle='--', alpha=0.7,
                label=f'Optimal: {opt_freq_phase/1000:.3f} kHz')
    
    # Mark the actual SNR at optimal frequency
    optimal_idx = np.argmin(np.abs(frequencies - opt_freq_phase))
    optimal_snr = snr_values[optimal_idx]
    ax5.plot(opt_freq_phase/1000, optimal_snr, 'ro', markersize=8, 
             label=f'SNR: {optimal_snr:.1f} dB')
    
    ax5.set_xlabel('Frequency (kHz)')
    ax5.set_ylabel('SNR (dB)')
    ax5.set_title('5. SNR vs Frequency', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # =======================================================================
    # PANEL 6: Residual Noise Analysis
    # =======================================================================
    ax6 = plt.subplot(3, 3, 6)
    
    residual_steady = residual[steady_start:]
    time_residual = time_ms[steady_start:]
    
    # Time domain residual
    ax6.plot(time_residual, residual_steady*1000, 'k-', alpha=0.7, linewidth=0.5)
    ax6.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    ax6.axhline(y=np.std(residual_steady)*1000, color='blue', linestyle='--', alpha=0.7,
                label=f'+1σ: {np.std(residual_steady)*1000:.3f} mV')
    ax6.axhline(y=-np.std(residual_steady)*1000, color='blue', linestyle='--', alpha=0.7,
                label=f'-1σ: {-np.std(residual_steady)*1000:.3f} mV')
    
    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Residual Noise (mV)')
    ax6.set_title(f'6. Residual Noise\n(Std: {np.std(residual_steady)*1000:.3f} mV)', 
                 fontweight='bold', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # =======================================================================
    # PANEL 7: XYRT Components Over Time
    # =======================================================================
    ax7 = plt.subplot(3, 3, 7)
    
    # Plot X, Y, R components
    ax7.plot(time_ms, X*1000, 'b-', linewidth=1.5, label='X (In-phase)', alpha=0.8)
    ax7.plot(time_ms, Y*1000, 'r-', linewidth=1.5, label='Y (Quadrature)', alpha=0.8)
    ax7.plot(time_ms, R*1000, 'g-', linewidth=1.5, label='R (Magnitude)', alpha=0.8)
    
    # Add steady-state region indication
    ax7.axvline(x=time_ms[steady_start], color='gray', linestyle=':', alpha=0.7,
                label='Steady-State Start')
    
    ax7.set_xlabel('Time (ms)')
    ax7.set_ylabel('Amplitude (mV)')
    ax7.set_title('7. XYRT Components Over Time', fontweight='bold', fontsize=12)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # =======================================================================
    # PANEL 8: Phase Component Over Time
    # =======================================================================
    ax8 = plt.subplot(3, 3, 8)
    
    # Plot phase component
    ax8.plot(time_ms, np.degrees(theta), 'purple', linewidth=1.5, alpha=0.8)
    
    # Add steady-state phase line
    steady_phase = np.degrees(np.mean(theta[steady_start:]))
    ax8.axhline(y=steady_phase, color='red', linestyle='--', linewidth=2,
                label=f'Steady State: {steady_phase:.1f}°')
    
    # Add target phase line
    ax8.axhline(y=input_params['phase_offset'], color='green', linestyle=':', linewidth=2,
                label=f'Target: {input_params["phase_offset"]}°')
    
    # Add steady-state region indication
    ax8.axvline(x=time_ms[steady_start], color='gray', linestyle=':', alpha=0.7,
                label='Steady-State Start')
    
    ax8.set_xlabel('Time (ms)')
    ax8.set_ylabel('Phase (degrees)')
    ax8.set_title('8. Detected Phase Over Time', fontweight='bold', fontsize=12)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # =======================================================================
    # PANEL 9: Performance Summary
    # =======================================================================
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate final performance metrics
    final_snr = 10 * np.log10(np.var(reconstructed[steady_start:]) / np.var(residual[steady_start:]))
    settling_time = time_ms[steady_start]
    
    summary_text = (
        "PERFORMANCE SUMMARY\n\n"
        f"Target Signal:\n"
        f"  Freq: {input_params['signal_frequency']/1000:.1f} kHz\n"
        f"  Amp: {input_params['amplitude']*1000:.1f} mV\n"
        f"  Phase: {input_params['phase_offset']}°\n\n"
        f"Optimal Detection:\n"
        f"  Freq (X): {opt_freq_phase/1000:.3f} kHz\n"
        f"  Freq (R): {opt_freq_mag/1000:.3f} kHz\n"
        f"  Phase: {np.degrees(theta_steady):.1f}°\n\n"
        f"Final Performance:\n"
        f"  SNR: {final_snr:.1f} dB\n"
        f"  Residual: {np.std(residual_steady)*1000:.3f} mV\n"
        f"  Settling: {settling_time:.1f} ms"
    )
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             family='monospace')
    
    # Adjust layout and save
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'lockin_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'lockin_analysis_{timestamp}.jpg', dpi=150, bbox_inches='tight')
    plt.show()

def comprehensive_performance_analysis(args):
    """Comprehensive analysis with all requested metrics"""
    logger = setup_logging(f'lockin_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.info("="*70)
    logging.info("SOFTWARE LOCK-IN AMPLIFIER - COMPREHENSIVE PERFORMANCE ANALYSIS")
    logging.info("="*70)
    logging.info(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Input parameters from command line or defaults
    input_params = {
        'signal_frequency': args.frequency,
        'amplitude': args.amplitude,
        'duration': args.duration,
        'sampling_rate': args.sampling_rate,
        'time_constant': args.time_constant,
        'noise_level': args.noise_level,
        'phase_offset': args.phase_offset
    }
    
    logging.info("\nINPUT PARAMETERS:")
    for key, value in input_params.items():
        if 'frequency' in key:
            logging.info(f"  {key}: {value/1000:.1f} kHz")
        elif key == 'amplitude':
            logging.info(f"  {key}: {value*1000:.1f} mV")
        elif key == 'duration':
            logging.info(f"  {key}: {value*1000:.1f} ms")
        elif key == 'sampling_rate':
            logging.info(f"  {key}: {value/1e6:.1f} MHz")
        else:
            logging.info(f"  {key}: {value}")
    
    # Generate signal
    logging.info("\nGenerating test signal...")
    signal, time, f_target, clean = generate_realistic_signal(
        f_signal=input_params['signal_frequency'],
        amplitude=input_params['amplitude'],
        duration=input_params['duration'],
        noise_level=input_params['noise_level'],
        phase_offset=input_params['phase_offset']
    )
    
    # Initialize lock-in
    logging.info("Initializing Lock-In Amplifier...")
    lockin = OptimizedLockInAmplifier(
        fs=input_params['sampling_rate'], 
        time_constant=input_params['time_constant']
    )
    
    # 1. Broad frequency scan for comprehensive analysis
    logging.info("\n1. COMPREHENSIVE FREQUENCY SCAN")
    freq_range = (args.freq_start, args.freq_end)
    step_size = args.freq_step
    
    (opt_freq_phase, opt_freq_mag, frequencies, phase_responses, 
     magnitude_responses, execution_times, residual_noises) = lockin.scan_reference_frequency(
        signal, time, freq_range, step_size)
    
    # 2. Autophase at optimal frequency
    logging.info("\n2. AUTOPHASE OPTIMIZATION")
    optimal_phase = lockin.autophase(signal, time, opt_freq_phase)
    
    # 3. Final demodulation with detailed analysis
    logging.info("\n3. DETAILED DEMODULATION ANALYSIS")
    X, Y, R, theta = lockin.demodulate(signal, time, opt_freq_phase)
    
    # Calculate comprehensive metrics
    steady_start = int(len(X) * 0.5)
    x_steady = np.mean(X[steady_start:])
    y_steady = np.mean(Y[steady_start:])
    r_steady = np.mean(R[steady_start:])
    theta_steady = np.mean(theta[steady_start:])
    
    x_std = np.std(X[steady_start:])
    y_std = np.std(Y[steady_start:])
    r_std = np.std(R[steady_start:])
    theta_std = np.std(theta[steady_start:])
    
    # Calculate SNR and reconstruction quality
    reconstructed = R * np.cos(2 * np.pi * opt_freq_phase * time + theta)
    residual = signal - reconstructed
    signal_power = np.var(reconstructed[steady_start:])
    noise_power = np.var(residual[steady_start:])
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    
    # Error analysis
    true_amplitude = input_params['amplitude']
    x_expected = true_amplitude / np.sqrt(2)  # Theoretical X for in-phase
    x_error = abs(x_steady - x_expected) / x_expected * 100
    r_error = abs(r_steady - true_amplitude) / true_amplitude * 100
    phase_error = abs(np.degrees(theta_steady) - input_params['phase_offset'])
    
    # Log comprehensive results
    logging.info("\nFINAL RESULTS WITH UNCERTAINTIES:")
    logging.info(f"Optimal Frequency (Phase): {opt_freq_phase/1000:.3f} ± {step_size/2000:.3f} kHz")
    logging.info(f"Optimal Frequency (Magnitude): {opt_freq_mag/1000:.3f} ± {step_size/2000:.3f} kHz")
    logging.info(f"Optimal Phase: {np.degrees(optimal_phase):.1f}°")
    logging.info(f"X-component: {x_steady*1000:.3f} ± {x_std*1000:.3f} mV (Error: {x_error:.1f}%)")
    logging.info(f"Y-component: {y_steady*1000:.3f} ± {y_std*1000:.3f} mV")
    logging.info(f"R-magnitude: {r_steady*1000:.3f} ± {r_std*1000:.3f} mV (Error: {r_error:.1f}%)")
    logging.info(f"Phase: {np.degrees(theta_steady):.1f} ± {np.degrees(theta_std):.1f}° (Error: {phase_error:.1f}°)")
    logging.info(f"SNR: {snr_db:.1f} dB")
    logging.info(f"Residual Noise: {np.std(residual)*1000:.3f} mV")
    
    # Create comprehensive plots
    logging.info("\nGenerating comprehensive plots...")
    create_comprehensive_plots(signal, time, clean, X, Y, R, theta, reconstructed, residual,
                              frequencies, phase_responses, magnitude_responses, 
                              execution_times, residual_noises, opt_freq_phase, opt_freq_mag,
                              input_params)
    
    logging.info(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("Results saved to:")
    logging.info(f"  - Log file: lockin_analysis_*.log")
    logging.info(f"  - Plot files: lockin_analysis_*.png, lockin_analysis_*.jpg")
    
    return lockin, signal, time, opt_freq_phase

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description='Software Lock-In Amplifier - Comprehensive Performance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python software-LIA.py
  python software-LIA.py -f 100000 -a 0.005 -d 0.02
  python software-LIA.py --freq-start 50000 --freq-end 150000 --freq-step 250
        """
    )
    
    # Signal parameters
    parser.add_argument('-f', '--frequency', type=float, default=123000.0,
                       help='Signal frequency in Hz (default: 123000)')
    parser.add_argument('-a', '--amplitude', type=float, default=0.01,
                       help='Signal amplitude in V (default: 0.01)')
    parser.add_argument('-d', '--duration', type=float, default=0.01,
                       help='Signal duration in seconds (default: 0.01)')
    parser.add_argument('-p', '--phase-offset', type=float, default=45.0,
                       help='Signal phase offset in degrees (default: 45)')
    parser.add_argument('-n', '--noise-level', type=float, default=0.005,
                       help='Noise level (default: 0.005)')
    
    # Lock-in parameters
    parser.add_argument('-sr', '--sampling-rate', type=float, default=2e6,
                       help='Sampling rate in Hz (default: 2e6)')
    parser.add_argument('-tc', '--time-constant', type=float, default=0.001,
                       help='Lock-in time constant in seconds (default: 0.001)')
    
    # Frequency scan parameters
    parser.add_argument('--freq-start', type=float, default=80000.0,
                       help='Frequency scan start in Hz (default: 80000)')
    parser.add_argument('--freq-end', type=float, default=160000.0,
                       help='Frequency scan end in Hz (default: 160000)')
    parser.add_argument('--freq-step', type=float, default=500.0,
                       help='Frequency scan step size in Hz (default: 500)')
    
    args = parser.parse_args()
    
    print("Software Lock-In Amplifier - Starting Analysis...")
    print("="*50)
    
    try:
        lockin, signal, time, optimal_freq = comprehensive_performance_analysis(args)
        print("\n" + "="*50)
        print("Analysis completed successfully!")
        print("Check the generated log files and plots for results.")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":

    main()
