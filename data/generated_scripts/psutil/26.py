import datetime
import time
import os
import psutil
from unittest.mock import patch

def get_path_and_priority(process_id):
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
    with patch('psutil.Process') as mock_process:
        instance = mock_process.return_value
        instance.exe.return_value = mock_path
        instance.nice.return_value = mock_priority
        return get_path_and_priority(process_id)

def change_process_priority(process_id, new_priority):
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
    try:
        p = psutil.Process(process_id)
        p.terminate()
        print(f'Process with id {process_id} terminated')
    except psutil.NoSuchProcess:
        print(f'Process with id {process_id} does not exist')
    except psutil.AccessDenied:
        print(f'Access denied for process with id {process_id}')

def print_disk_io_stats():
    print('Disk I/O information:')
    io_info = psutil.disk_io_counters()
    attrs = ['read_count', 'write_count', 'read_bytes', 'write_bytes', 'read_time', 'write_time']
    for attr in attrs:
        print(f' {attr}: {getattr(io_info, attr)}')

def main():
    current_process_id = os.getpid()
    print(f'Path and priority of current process: {get_path_and_priority(current_process_id)}')
    print('Mocking path and priority:')
    print(mock_process_path_and_priority(9999, '/fake/path', 20))
    change_process_priority(current_process_id, 10)
    print_disk_io_stats()

if __name__ == "__main__":
    main()