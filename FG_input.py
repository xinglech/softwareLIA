"""
Simplified Function Generator Data Acquisition
Based on proven working code - extracts data from Rohde & Schwarz RTB2004 oscilloscope
With essential logging for debugging and monitoring
"""

from RsInstrument import *
import numpy as np
import logging
from datetime import datetime
import os

class FunctionGenerator:
    def __init__(self, ip_address: str = '192.168.0.5', enable_logging: bool = True):
        self.ip_address = ip_address
        self.instrument = None
        self.enable_logging = enable_logging
        self._setup_logging()
        self._connect()
    
    def _setup_logging(self):
        """Simple logging setup that won't fail"""
        if self.enable_logging:
            # Create a simple log file in current directory
            log_filename = f"fg_acquisition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filename, mode='w'),
                    logging.StreamHandler()  # Also print to console
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"FG Acquisition logging started: {log_filename}")
        else:
            # Create a null logger if logging is disabled
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
    
    def _connect(self):
        """Simple connection to the oscilloscope"""
        try:
            RsInstrument.assert_minimum_version('1.102.0')
            
            self.logger.info(f"Attempting connection to {self.ip_address}")
            
            # Simple connection with minimal logging
            self.instrument = RsInstrument(f'TCPIP::{self.ip_address}::INSTR', 
                                         options='LoggingMode=Off, RsInstrumentTimeout=5000')
            
            # Test basic communication
            idn = self.instrument.query('*IDN?')
            self.logger.info(f"Connected to: {idn.strip()}")
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.instrument = None
            raise
    
    def configure_oscilloscope(self):
        """
        Configure oscilloscope with minimal, proven commands
        Based on colleague's working settings
        """
        if not self.instrument:
            self.logger.warning("No instrument connection for configuration")
            return
        
        try:
            # Essential configuration commands that work with RTB2004
            config_commands = [
                'CHAN:TYPE HRES',      # Set high resolution mode (16-bit data)
                'TIM:SCAL 1E-7',       # Set time base
                'FORM REAL',           # Set data format to REAL (32-bit)  
                'FORM:BORD LSBF',      # Set little endian byte order
                'CHAN:DATA:POIN DMAX', # Collect max displayed points
                'TRIG:A:MODE AUTO',    # Auto trigger mode
            ]
            
            self.logger.info("Configuring oscilloscope...")
            
            for cmd in config_commands:
                try:
                    self.instrument.write(cmd)
                    self.logger.debug(f"Sent command: {cmd}")
                except Exception as e:
                    self.logger.warning(f"Command failed {cmd}: {e}")
                    # Continue with other commands
            
            self.logger.info("Oscilloscope configuration completed")
            
        except Exception as e:
            self.logger.error(f"Configuration failed: {e}")
            # Continue anyway - some commands might not be critical
    
    def acquire_data(self, duration: float = 0.01) -> tuple:
        """
        Simple data acquisition - the core function
        Returns: (signal_data, time_array, sampling_rate)
        """
        if not self.instrument:
            self.logger.error("No instrument connection for data acquisition")
            raise ConnectionError("No instrument connection")
        
        try:
            self.logger.info("Starting data acquisition...")
            
            # Single acquisition command
            self.instrument.write('SING')
            self.logger.debug("Sent SING command")
            
            # Wait briefly for acquisition
            import time
            time.sleep(0.5)
            
            # Request the data - using the exact command from working code
            self.logger.debug("Querying CHAN:DATA...")
            datastr = self.instrument.query('CHAN:DATA?')
            self.logger.debug(f"Received data string length: {len(datastr)}")
            
            # Convert to numpy array - same as working code
            signal_data = np.array(datastr.split(","), dtype=np.float32)
            
            # Create time array based on duration
            actual_samples = len(signal_data)
            time_array = np.linspace(0, duration, actual_samples)
            
            # Calculate actual sampling rate
            sampling_rate = actual_samples / duration
            
            # Log acquisition results
            self.logger.info(f"Acquisition successful: {actual_samples} samples at {sampling_rate/1e6:.2f} MHz")
            self.logger.info(f"Signal statistics: min={np.min(signal_data):.6f}, max={np.max(signal_data):.6f}, mean={np.mean(signal_data):.6f}")
            
            return signal_data, time_array, sampling_rate
            
        except Exception as e:
            self.logger.error(f"Data acquisition failed: {e}")
            raise
    
    def get_instrument_info(self) -> dict:
        """Get basic instrument information"""
        if not self.instrument:
            return {"error": "No connection"}
        
        try:
            info = {
                'idn': self.instrument.query('*IDN?').strip(),
                'timebase': self.instrument.query('TIM:SCAL?').strip(),
            }
            self.logger.info(f"Instrument info: {info}")
            return info
        except Exception as e:
            self.logger.warning(f"Could not get instrument info: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Cleanup with logging"""
        if self.instrument:
            self.instrument.close()
            self.logger.info("Instrument connection closed")
        else:
            self.logger.info("No instrument connection to close")

# Simple acquisition function for backward compatibility
def acquire_fg_data(ip_address: str = '192.168.0.5', 
                   duration: float = 0.01, 
                   sampling_rate: float = 2e6,
                   enable_logging: bool = True) -> tuple:
    """
    Simple function to acquire data - main interface for softwareLIA.py
    Maintains same signature as original
    """
    fg = FunctionGenerator(ip_address, enable_logging)
    
    try:
        fg.configure_oscilloscope()
        signal, time_arr, actual_fs = fg.acquire_data(duration)
        return signal, time_arr, actual_fs
        
    finally:
        fg.close()

# Enhanced version with fallback for softwareLIA.py
def acquire_fg_data_robust(ip_address: str = '192.168.0.5', 
                          duration: float = 0.01, 
                          sampling_rate: float = 2e6,
                          enable_logging: bool = True) -> tuple:
    """
    Robust version with fallback for softwareLIA.py
    Returns: (signal, time, actual_fs, connection_type)
    """
    try:
        signal, time_arr, actual_fs = acquire_fg_data(ip_address, duration, sampling_rate, enable_logging)
        if enable_logging:
            logging.info(f"Hardware acquisition successful via TCP")
        return signal, time_arr, actual_fs, 'HARDWARE'
        
    except Exception as e:
        if enable_logging:
            logging.error(f"Hardware acquisition failed: {e}, using simulated data")
        # Fallback to simulated data
        signal, time_arr, actual_fs = _generate_simulated_signal(duration, sampling_rate)
        return signal, time_arr, actual_fs, 'SIMULATED'

def _generate_simulated_signal(duration: float = 0.01, sampling_rate: float = 2e6) -> tuple:
    """Generate simulated data as fallback"""
    samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, samples)
    
    # Simple simulated signal
    main_freq = 100000  # 100 kHz
    amplitude = 0.01
    
    clean_signal = amplitude * np.sin(2 * np.pi * main_freq * t)
    noise = 0.005 * np.random.randn(samples)
    
    simulated_signal = clean_signal + noise
    
    logging.info(f"Generated simulated signal: {samples} samples at {sampling_rate/1e6:.1f} MHz")
    
    return simulated_signal, t, sampling_rate

# Test function
if __name__ == "__main__":
    print("Testing simplified FG acquisition with logging...")
    
    try:
        # Test basic acquisition
        signal, time_arr, fs = acquire_fg_data()
        print(f"Success: Acquired {len(signal)} samples")
        print(f"Signal stats: min={np.min(signal):.6f}, max={np.max(signal):.6f}, mean={np.mean(signal):.6f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
