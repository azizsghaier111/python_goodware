import datetime
from unittest.mock import patch
import psutil
import time
import os 

def get_process_info(process_id):
    try:
        p = psutil.Process(process_id)
        details = {...}

        # Get process priority
        try:
            details['priority'] = p.nice()
        except psutil.AccessDenied:
            details['priority'] = "Access Denied"
            
    except psutil.NoSuchProcess:
        details = None

    return details

def print_process_info(details, indent=0):
    # the same as before

    # Add print for priority
    print(" " * indent, "Priority:", details["priority"])

def change_process_priority(process_id, priority):
    try:
        p = psutil.Process(process_id)
        p.nice(priority)
    except psutil.NoSuchProcess:
        print("There is no process with pid:", process_id)
    except psutil.AccessDenied:
        print("Access denied to change priority of process with pid:", process_id)

def terminate_process(process_id):
    try:
        p = psutil.Process(process_id)
        p.terminate()
    except psutil.NoSuchProcess:
        print("There is no process with pid:", process_id)
    except psutil.AccessDenied:
        print("Access denied to terminate process with pid:", process_id)

def print_disk_io_stats():
    disk_io = psutil.disk_io_counters()
    print("Disk I/O statistics:")
    print(" Read count:", disk_io.read_count)
    print(" Write count:", disk_io.write_count)
    print(" Read bytes:", disk_io.read_bytes)
    print(" Write bytes:", disk_io.write_bytes)
    print(" Read time:", disk_io.read_time, "ms")
    print(" Write time:", disk_io.write_time, "ms")

def main():
    print_process_info(get_process_info(None))
    print_disk_io_stats()

    # You can change and terminate process here and make sure you have access to it
    # change_process_priority(None, 10)
    # time.sleep(10)
    # terminate_process(None)

if __name__ == "__main__":
    main()