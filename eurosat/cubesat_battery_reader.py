# -*- coding: utf-8 -*-
"""
CubeSat Battery Reader for Federated Learning
Connects to physical CubeSat and reads battery level
"""

import serial
import time
from typing import Tuple, Optional


_cubesat_connection = None
_serial_port = None
_baud_rate = None


def initialize_cubesat(serial_port: str = "/dev/cu.usbserial-1110", baud_rate: int = 9600):
    """
    Initialize connection to CubeSat
    
    Args:
        serial_port: Serial port path
        baud_rate: Baud rate for serial communication
    """
    global _cubesat_connection, _serial_port, _baud_rate
    
    _serial_port = serial_port
    _baud_rate = baud_rate
    
    try:
        _cubesat_connection = serial.Serial(serial_port, baud_rate, timeout=2)
        time.sleep(0.5)  # Let connection stabilize
        
        # Flush any stale data in buffer
        _cubesat_connection.reset_input_buffer()
        time.sleep(0.2)
        
        # Do a test read to verify connection
        test_battery, test_operational = read_cubesat_battery()
        if test_battery > 0:
            print(f"✅ Connected to CubeSat on {serial_port} (Battery: {test_battery:.1f}%)")
        else:
            print(f"✅ Connected to CubeSat on {serial_port} (waiting for data...)")
        
        return True
    except Exception as e:
        print(f"⚠️  Could not connect to CubeSat: {e}")
        _cubesat_connection = None
        return False


def read_cubesat_battery() -> Tuple[float, bool]:
    """
    Read battery level from CubeSat with retry logic
    
    Returns:
        Tuple of (battery_percentage, operational)
    """
    global _cubesat_connection
    
    if _cubesat_connection is None:
        return 0.0, False
    
    # Try up to 5 times to get valid data
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Read a line from serial
            line = _cubesat_connection.readline()
            
            if not line:
                time.sleep(0.1)
                continue
            
            # Decode and parse
            response = line.decode('utf-8', errors='ignore').strip()
            
            # Skip non-data lines
            if not response or ',' not in response:
                continue
            
            # Parse CSV (battery is at index 12, voltage at 13)
            parts = response.split(',')
            
            if len(parts) >= 13:
                battery = float(parts[12])  # 13th value is battery %
                voltage = float(parts[13]) if len(parts) > 13 else 0
                
                # Operational if battery reading is valid
                operational = battery > 0
                
                if battery > 0:  # Valid reading
                    return battery, operational
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"⚠️  Error reading CubeSat battery: {e}")
            continue
    
    # If all retries failed
    return 0.0, False


def cleanup_cubesat():
    """Disconnect from CubeSat"""
    global _cubesat_connection
    
    if _cubesat_connection is not None:
        try:
            _cubesat_connection.close()
            print("🔌 Disconnected from CubeSat")
        except:
            pass
        _cubesat_connection = None

