import time
import os
import datetime
import psutil
from unittest.mock import patch

def get_path_and_priority(process_id):
    """
    Given a process id, this function returns the path 
    and the priority of the process
    """
    try:
        p = psutil.Process(process_id)
        path = p.exe()
        priority = p.nice()
        return {'path': path, 'priority': priority}
    except psutil.NoSuchProcess:
        print(f'Process with id {process_id} does not exist')
    except psutil.AccessDenied:
        print(f'Access denied for process with id {process_id}')

def mock_process_path_and_priority(process_id, mock_path, mock_priority):
    """
    This function mock a process with pseudo path and priority
    """
    with patch('psutil.Process') as mock_process:
        instance = mock_process.return_value
        instance.exe.return_value = mock_path
        instance.nice.return_value = mock_priority
        return get_path_and_priority(process_id)

def change_process_priority(process_id, new_priority):
    """
    Change the priority of a process with given id
    """
    try:
        p = psutil.Process(process_id)
        old_priority = p.nice()
        p.nice(new_priority)
        print(f'Priority changed for process with id {process_id} from {old_priority} to {new_priority}')
    except psutil.NoSuchProcess:
        print(f'Process with id {process_id} does not exist')
    except psutil.AccessDenied:
        print(f'Access denied for process with id {process_id}')

def terminate_process(process_id):
    """
    Terminate a process with a given process id
    """
    try:
        p = psutil.Process(process_id)
        p.terminate()
        print(f'Process with id {process_id} terminated')
    except psutil.NoSuchProcess:
        print(f'Process with id {process_id} does not exist')
    except psutil.AccessDenied:
        print(f'Access denied for process with id {process_id}')

def print_disk_io_stats():
    """
    Output Disk I/O info
    """
    print('Disk I/O information:')
    io_info = psutil.disk_io_counters()
    attrs = ['read_count', 'write_count', 'read_bytes', 'write_bytes', 'read_time', 'write_time']
    for attr in attrs:
        print(f'  {attr}: {getattr(io_info, attr)}')

def memory_info():
    """
    Output memory statistics
    """
    print("Memory Information:")
    mem_info = psutil.virtual_memory()
    attrs = ['total', 'available', 'percent', 'used', 'free']
    for attr in attrs:
        print(f'  {attr}: {getattr(mem_info, attr)}')

def print_cpu_percent():
    """
    Output current CPU utilization
    """
    print("Current CPU utilization is {}%".format(psutil.cpu_percent(interval=1)))

def print_every_process():
    """
    Output every process running in the system
    """
    print("All processes running in the system:")
    for proc in psutil.process_iter(['pid', 'name']):
        print(proc.info)

def main():
    # Getting process id of the current running process
    current_process_id = os.getpid()

    print(f'Path and priority of current process: {get_path_and_priority(current_process_id)}')

    print('Mocking path and priority:')
    print(mock_process_path_and_priority(9999, '/fake/path', 20))

    # Changing the priority of the current process
    change_process_priority(current_process_id, 10)

    # Printing Dis IO stats
    print_disk_io_stats()

    # Get Memory Information
    memory_info()

    # Get CPU Usage
    print_cpu_percent()

    # Get every running process
    print_every_process()

    # Wait for 5 seconds
    time.sleep(5)

    # Terminate the current process
    terminate_process(current_process_id)

if __name__ == "__main__":
    main()