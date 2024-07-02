import psutil
import datetime
import time
import numpy as np
from unittest.mock import patch, MagicMock

# System Memory Information functionality
def get_memory_info():
    """
    This function returns the system's memory usage details.
    :return: dict: System's memory details
    """
    mem_info = psutil.virtual_memory()
    return {
        "total": mem_info.total,
        "available": mem_info.available,
        "percent": mem_info.percent,
        "used": mem_info.used,
        "active": mem_info.active
    }

# CPU Sensor Information functionality
def get_cpu_temp():
    """
    This function returns the CPU's sensor information.
    :return: float: CPU temperature
    """
    temps = psutil.sensors_temperatures()
    return temps['cpu-thermal'][0].current if 'cpu-thermal' in temps else None

#Get process's children
def get_child_processes(process_id):
    """
    This function returns the child processes for given process id.
    :param process_id: process id to get child processes
    :return: list: child processes of the process
    """
    try:
        process = psutil.Process(process_id)
        return process.children()
    except psutil.NoSuchProcess:
        return None
        
def mock_functions():
    """
    This function demonstrates the use of the 'mock' library to mock functionalities for testing.
    """
    with patch('psutil.virtual_memory') as mock_memory:
        mock_memory.return_value = MagicMock(
            total=1000, available=500, percent=50.0, used=500, active=200)
        print("Mock Memory Information: ", get_memory_info())

    with patch('psutil.sensors_temperatures') as mock_temps:
        mock_temps.return_value = {'cpu-thermal': [MagicMock(current=45.0)]}
        print("Mock CPU Temperature: ", get_cpu_temp())

    with patch('psutil.Process.children') as mock_children:
        mock_children.return_value = [MagicMock()]
        print("Mock Child Processes: ", get_child_processes(1))

def main():
    print("System uptime: ", get_system_uptime())
    print("Boot Time: ", get_boot_time())
    print("Threads for process id 1: ", get_process_threads(1))
    print("Memory information: ", get_memory_info())
    print("CPU temperature: ", get_cpu_temp())
    print("Child processes for process id 1: ", get_child_processes(1))

    print("\nRunning mock functions...")
    mock_functions()

if __name__ == "__main__":
    main()