import psutil          
import datetime         
import time             
from unittest.mock import patch
import socket

# Sensor-related functionality
def get_sensor_info():
    try:
        sensors = psutil.sensors_temperatures()
    except Exception as e:
        sensors = str(e)
    return sensors

# Network Input/Output functionality
def get_net_io_count():
    try:
        net_io = psutil.net_io_counters(pernic=True)
    except Exception as e:
        net_io = str(e)
    return net_io

#System uptime functionality
def get_system_uptime():
    try:
        uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    except Exception as e:
        uptime = str(e)
    return uptime

# System boot time functionality
def get_boot_time():
    try:
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
    except Exception as e:
        boot_time = str(e)
    return boot_time

# Process-related functionality
def get_process_threads(process_id):
    try:
        process = psutil.Process(process_id)
        process_info = process.threads()
    except psutil.NoSuchProcess:
        process_info = "No Such Process"
    return process_info

def main():
    print("System Uptime: ", get_system_uptime())
    print("System Boot Time: ", get_boot_time())
    print("\nNetwork I/O statistics:\n")
    print(get_net_io_count())
    print("\nSensor Info:\n")
    print(get_sensor_info())
    print("\nProcess threads for process with id 1:\n")
    print(get_process_threads(1))

# Mocking for unit testing
def mock_uptime_and_boottime():
    with patch('psutil.boot_time', return_value=time.time()):
        print("System Uptime: ", get_system_uptime())
        print("System Boot Time: ", get_boot_time())

def mock_net_io_count():
    with patch('psutil.net_io_counters', 
            return_value={'lo': psutil._common.snetio(bytes_sent=0, bytes_recv=0, packets_sent=0, 
            packets_recv=0, errin=0, errout=0, dropin=0, dropout=0)}):
        print("\nNetwork I/O statistics:\n")
        print(get_net_io_count())

def mock_sensor_info():
    with patch('psutil.sensors_temperatures', 
            return_value={'coretemp': 
            [psutil._common.sensors_temperatures_fahrenheit(label='Package id 0', 
            current=38.0, high=100.0, critical=100.0)]}):
        print("\nSensor Info:\n")
        print(get_sensor_info())

def mock_process_info():
    with patch('psutil.Process', return_value=psutil.Process(1)):
        print("\nProcess threads for process with id 1:\n")
        print(get_process_threads(1))

if __name__ == "__main__":
    main()
    print("\n----- Mocking equations for validation -----\n")
    mock_uptime_and_boottime()
    mock_net_io_count()
    mock_sensor_info()
    mock_process_info()