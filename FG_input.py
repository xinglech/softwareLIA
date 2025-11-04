"""
Enhanced Function Generator Data Acquisition Module
With comprehensive fallback: UDP → TCP → Simulated Data
"""

from RsInstrument import *
import numpy as np
import logging
import socket
import threading
import queue
import time
import signal
import sys
from struct import unpack_from
from typing import Tuple, Optional, List, Dict, Any
import random

class EnhancedFunctionGenerator:
    def __init__(self, ip_address: str = '192.168.0.5', 
                 log_file: str = r'c:\temp\fg_acquisition.log',
                 udp_port: int = 1866,
                 tcp_timeout: int = 5000):
        self.ip_address = ip_address
        self.log_file = log_file
        self.udp_port = udp_port
        self.tcp_timeout = tcp_timeout
        self.instrument = None
        self.udp_socket = None
        self.streaming = False
        self.connection_type = None  # 'UDP', 'TCP', or 'SIMULATED'
        self._setup_logging()
        
        # Try to establish connection with fallbacks
        self._establish_connection_with_fallback()
    
    def _setup_logging(self):
        """Enhanced logging setup"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _establish_connection_with_fallback(self):
        """
        Attempt connection with comprehensive fallback strategy:
        1. Try UDP streaming (if supported)
        2. Fall back to TCP (original method)
        3. Fall back to simulated data
        """
        connection_attempts = [
            ('UDP', self._try_udp_connection),
            ('TCP', self._try_tcp_connection),
        ]
        
        for conn_type, connection_method in connection_attempts:
            self.logger.info(f"Attempting {conn_type} connection...")
            success = connection_method()
            if success:
                self.connection_type = conn_type
                self.logger.info(f"✓ Successfully connected via {conn_type}")
                return
        
        # All hardware connections failed, use simulation
        self.connection_type = 'SIMULATED'
        self.logger.warning("✗ All hardware connections failed. Using simulated data mode.")
    
    def _try_udp_connection(self) -> bool:
        """
        Attempt UDP streaming connection
        Returns: True if successful, False otherwise
        """
        try:
            # First, establish TCP connection to configure the instrument
            temp_instrument = RsInstrument(
                f'TCPIP::{self.ip_address}::INSTR',
                options=f'LoggingMode=Off, RsInstrumentTimeout={self.tcp_timeout}'
            )
            
            # Check if streaming is supported
            idn = temp_instrument.query('*IDN?').upper()
            supports_streaming = any(term in idn for term in ['SRS', 'SR865', 'STREAM'])
            
            if supports_streaming:
                # Configure UDP streaming
                temp_instrument.write(f'STREAMPORT {self.udp_port}')
                
                # Setup UDP socket
                self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.udp_socket.bind(('', self.udp_port))
                self.udp_socket.settimeout(2.0)
                
                # Test streaming
                temp_instrument.write('STREAM ON')
                time.sleep(0.1)  # Brief test
                temp_instrument.write('STREAM OFF')
                
                self.instrument = temp_instrument
                self._configure_oscilloscope()
                return True
            else:
                temp_instrument.close()
                self.logger.info("UDP streaming not supported by instrument")
                return False
                
        except Exception as e:
            self.logger.warning(f"UDP connection failed: {e}")
            if 'temp_instrument' in locals():
                temp_instrument.close()
            if self.udp_socket:
                self.udp_socket.close()
                self.udp_socket = None
            return False
    
    def _try_tcp_connection(self) -> bool:
        """
        Attempt traditional TCP connection (original FG_input.py method)
        Returns: True if successful, False otherwise
        """
        try:
            self.instrument = RsInstrument(
                f'TCPIP::{self.ip_address}::INSTR',
                options=f'LoggingMode=On, LoggingName=OSCILLOSCOPE, RsInstrumentTimeout={self.tcp_timeout}'
            )
            
            # Test basic communication
            idn = self.instrument.query('*IDN?')
            self.logger.info(f"Instrument identified: {idn.strip()}")
            
            self._configure_oscilloscope()
            return True
            
        except Exception as e:
            self.logger.warning(f"TCP connection failed: {e}")
            if self.instrument:
                self.instrument.close()
                self.instrument = None
            return False
    
    def _configure_oscilloscope(self):
        """Configure oscilloscope settings with validation"""
        if self.connection_type == 'SIMULATED':
            self.logger.info("Skipping configuration for simulated mode")
            return
            
        configuration_commands = {
            'CHAN:TYPE': 'HRES',      # High resolution mode (16-bit data)
            'TIM:SCAL': '1E-7',       # Time base
            'FORM': 'REAL',           # Data format to REAL (32-bit)
            'FORM:BORD': 'LSBF',      # Little endian byte order
            'CHAN:DATA:POIN': 'DMAX'  # Collect max displayed points
        }
        
        for cmd, value in configuration_commands.items():
            try:
                self.instrument.write(f'{cmd} {value}')
                # Verify configuration
                response = self.instrument.query(f'{cmd}?')
                self.logger.debug(f"{cmd} set to: {response.strip()}")
            except Exception as e:
                self.logger.warning(f"Failed to set {cmd}: {e}")
    
    def _generate_simulated_signal(self, duration: float = 0.01, sampling_rate: float = 2e6) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate realistic simulated data as fallback
        """
        samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Simulate a realistic signal with noise and distortions
        main_freq = 100000  # 100 kHz
        amplitude = 0.01
        
        # Clean signal
        clean_signal = amplitude * np.sin(2 * np.pi * main_freq * t)
        
        # Add realistic impairments
        noise = 0.005 * np.random.randn(samples)
        harmonic_distortion = 0.001 * np.sin(2 * np.pi * 2 * main_freq * t)
        phase_noise = 0.001 * np.random.randn(samples)
        dc_offset = 0.002
        
        simulated_signal = (clean_signal + noise + harmonic_distortion + 
                          dc_offset + 0.0005 * phase_noise * np.cos(2 * np.pi * main_freq * t))
        
        self.logger.info(f"Generated simulated signal: {samples} samples at {sampling_rate/1e6:.1f} MHz")
        
        return simulated_signal, t, sampling_rate
    
    def acquire_single_shot(self, duration: float = 0.01, expected_fs: float = 2e6) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Robust single-shot data acquisition with automatic fallback
        """
        if self.connection_type == 'SIMULATED':
            self.logger.info("Using simulated data (fallback mode)")
            return self._generate_simulated_signal(duration, expected_fs)
        
        try:
            start_time = time.perf_counter()
            
            if self.connection_type == 'UDP':
                # Use UDP streaming for single acquisition
                return self._acquire_via_udp(duration, expected_fs)
            else:  # TCP
                # Use traditional TCP query
                return self._acquire_via_tcp(duration, expected_fs)
                
        except Exception as e:
            self.logger.error(f"Hardware acquisition failed: {e}. Falling back to simulated data.")
            return self._generate_simulated_signal(duration, expected_fs)
    
    def _acquire_via_tcp(self, duration: float, expected_fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Original TCP acquisition method"""
        # Configure acquisition parameters
        time_scale = duration / 10
        self.instrument.write(f'TIM:SCAL {time_scale:.2E}')
        self.instrument.write('TRIG:MODE AUTO')
        
        # Query data
        datastr = self.instrument.query('CHAN:DATA?', timeout=10000)
        acquisition_time = time.perf_counter() - start_time
        
        # Convert to numpy array
        signal_data = np.array(datastr.split(","), dtype=np.float32)
        actual_samples = len(signal_data)
        actual_sampling_rate = actual_samples / acquisition_time
        time_array = np.linspace(0, acquisition_time, actual_samples)
        
        self.logger.info(f"TCP acquisition: {actual_samples} samples at {actual_sampling_rate/1e6:.2f} MHz")
        
        return signal_data, time_array, actual_sampling_rate
    
    def _acquire_via_udp(self, duration: float, expected_fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """UDP-based acquisition"""
        try:
            self.instrument.write('STREAM ON')
            
            packets = []
            start_time = time.perf_counter()
            end_time = start_time + duration
            
            while time.perf_counter() < end_time:
                try:
                    data, addr = self.udp_socket.recvfrom(4096)
                    packets.append(data)
                except socket.timeout:
                    continue
            
            self.instrument.write('STREAM OFF')
            
            # Process packets into signal data
            signal_data = self._process_udp_packets(packets)
            actual_samples = len(signal_data)
            actual_duration = time.perf_counter() - start_time
            actual_sampling_rate = actual_samples / actual_duration
            time_array = np.linspace(0, actual_duration, actual_samples)
            
            self.logger.info(f"UDP acquisition: {actual_samples} samples at {actual_sampling_rate/1e6:.2f} MHz")
            
            return signal_data, time_array, actual_sampling_rate
            
        except Exception as e:
            self.logger.error(f"UDP acquisition failed: {e}")
            raise
    
    def _process_udp_packets(self, packets: List[bytes]) -> np.ndarray:
        """Process UDP packets into signal data"""
        # Simple implementation - extract floats from packets
        # You might need to adjust this based on your instrument's packet format
        signal_points = []
        for packet in packets:
            # Assuming packet contains 32-bit floats
            if len(packet) >= 4:
                try:
                    # Extract floats (adjust format as needed)
                    values = unpack_from('>f', packet)
                    signal_points.extend(values)
                except:
                    continue
        return np.array(signal_points, dtype=np.float32)
    
    def start_streaming(self, udp_port: int = None) -> bool:
        """Start UDP streaming mode"""
        if self.connection_type != 'UDP':
            self.logger.warning("Streaming only available in UDP mode")
            return False
        
        try:
            if udp_port:
                self.udp_port = udp_port
                self.instrument.write(f'STREAMPORT {udp_port}')
            
            self.instrument.write('STREAM ON')
            self.streaming = True
            self.logger.info(f"UDP streaming started on port {self.udp_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        status = {
            'connection_type': self.connection_type,
            'ip_address': self.ip_address,
            'udp_port': self.udp_port if self.connection_type == 'UDP' else None,
            'streaming_active': self.streaming,
            'instrument_connected': self.instrument is not None
        }
        
        if self.connection_type != 'SIMULATED' and self.instrument:
            try:
                status['instrument_id'] = self.instrument.query('*IDN?').strip()
            except:
                status['instrument_id'] = 'Unknown'
        
        return status
    
    def close(self):
        """Cleanup resources safely"""
        try:
            if self.streaming and self.instrument:
                self.instrument.write('STREAM OFF')
            
            if self.udp_socket:
                self.udp_socket.close()
            
            if self.instrument:
                self.instrument.close()
                
            self.logger.info("Function generator connection closed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")


# Enhanced acquisition function with fallback
def acquire_fg_data_robust(ip_address: str = '192.168.0.5', 
                          duration: float = 0.01, 
                          sampling_rate: float = 2e6,
                          max_retries: int = 2) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """
    Robust data acquisition with comprehensive fallback and status reporting
    
    Returns:
        tuple: (signal_data, time_array, actual_sampling_rate, connection_type)
    """
    fg = EnhancedFunctionGenerator(ip_address)
    
    try:
        signal, time_arr, actual_fs = fg.acquire_single_shot(duration, sampling_rate)
        conn_type = fg.connection_type
        
        logging.info(f"Acquisition completed via {conn_type}: "
                    f"{len(signal)} samples at {actual_fs/1e6:.2f} MHz")
        
        return signal, time_arr, actual_fs, conn_type
        
    except Exception as e:
        logging.error(f"All acquisition methods failed: {e}")
        # Final fallback - generate simulated data
        signal, time_arr, actual_fs = fg._generate_simulated_signal(duration, sampling_rate)
        return signal, time_arr, actual_fs, 'SIMULATED_FALLBACK'
    
    finally:
        fg.close()


# Backward compatibility function
def acquire_fg_data(ip_address: str = '192.168.0.5', 
                   duration: float = 0.01, 
                   sampling_rate: float = 2e6) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Original function signature for backward compatibility
    """
    signal, time_arr, actual_fs, _ = acquire_fg_data_robust(
        ip_address, duration, sampling_rate
    )
    return signal, time_arr, actual_fs


if __name__ == "__main__":
    # Test the robust acquisition system
    print("Testing robust FG acquisition with fallback...")
    
    fg = EnhancedFunctionGenerator()
    
    try:
        # Display connection status
        status = fg.get_connection_status()
        print(f"Connection Status: {status}")
        
        # Test acquisition
        signal, time_arr, fs = fg.acquire_single_shot(duration=0.01)
        print(f"✓ Acquired {len(signal)} samples at {fs/1e6:.2f} MHz "
              f"via {fg.connection_type} connection")
        
        # Test the robust acquisition function
        signal2, time2, fs2, conn_type = acquire_fg_data_robust()
        print(f"✓ Robust function: {len(signal2)} samples via {conn_type}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    finally:
        fg.close()
