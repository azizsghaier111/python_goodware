import psutil
import datetime
import time
from unittest.mock import patch

def get_cpu_info():
    try:
        cpu_info = {'Physical cores': psutil.cpu_count(logical=False),
                    'Total cores': psutil.cpu_count(logical=True),
                    'Max Frequency (MHz)': psutil.cpu_freq().max,
                    'Min Frequency (MHz)': psutil.cpu_freq().min,
                    'Current Frequency (MHz)': psutil.cpu_freq().current,
                    'Total CPU Usage (%)': psutil.cpu_percent(),
                    'CPU Usage Per Core (%)': psutil.cpu_percent(percpu=True)}
    except Exception as e:
        cpu_info = str(e)
    return cpu_info

def get_memory_details():
    try:
        svmem = psutil.virtual_memory()
        memory_details = {'Total (GB)': svmem.total / (1024. ** 3),
                          'Available (GB)': svmem.available / (1024. ** 3),
                          'Used (GB)': svmem.used / (1024. ** 3),
                          'Memory Percentage (%)': svmem.percent}
    except Exception as e:
        memory_details = str(e)
    return memory_details

def main():
    print("\nCPU Info:\n")
    for k, v in get_cpu_info().items():
        print(f"{k}: {v}")
    print("\nMemory Details:\n")
    for k, v in get_memory_details().items():
        print(f"{k}: {v}")
    print("System Uptime: ", get_system_uptime())
    print("System Boot Time: ", get_boot_time())
    print("\nNetwork I/O statistics:\n")
    print(get_net_io_count())
    print("\nSensor Info:\n")
    print(get_sensor_info())
    print("\nProcess threads for process with id 1:\n")
    print(get_process_threads(1))

# Mocking for unit testing
def mock_cpu_info():
    with patch('psutil.cpu_count', return_value=4), \
            patch('psutil.cpu_freq', return_value=psutil._common.scpufreq(current=2200.0, min=0.0, max=2200.0)), \
            patch('psutil.cpu_percent', return_value=5.6):
        print("\nMocked CPU Info:\n") 
        for k, v in get_cpu_info().items():
            print(f"{k}: {v}")

def mock_memory_details():
    with patch('psutil.virtual_memory', 
               return_value=psutil._common.svmem(total=16726466560, available=6253568000, percent=62.6, used=10387521536,
                                                 free=609490944, active=8289890304, inactive=5153325056, buffers=769433600, 
                                                 cached=6153039872, shared=1016838144, slab=368144384)):
        print("\nMocked Memory Details:\n")
        for k, v in get_memory_details().items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
    print("\n----- Mocking equations for validation -----\n")
    mock_uptime_and_boottime()
    mock_net_io_count()
    mock_sensor_info()
    mock_cpu_info()
    mock_memory_details()
    mock_process_info()