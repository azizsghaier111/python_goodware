import psutil
import datetime
import time
import os
import numpy as np
from unittest.mock import patch
from pathlib import Path
from psutil._common import bytes2human
import torch
from pytorch_lightning import seed_everything

# Seeding for deterministic results
seed_everything(42)

def get_priority_class(process_id):
    """Function to return priority of the process"""
    try:
        process = psutil.Process(process_id)
        return process.nice()
    except psutil.NoSuchProcess:
        return None

def get_process_memory_info(process_id):
    """Function to get memory used by the process"""
    try:
        process = psutil.Process(process_id)
        info = process.memory_info()
        return {"rss": bytes2human(info.rss), "vms": bytes2human(info.vms)}
    except psutil.NoSuchProcess:
        return None

def get_process_exe_path(process_id):
    """Function to get process executable path"""
    try:
        process = psutil.Process(process_id)
        exe_path = process.exe()
        return exe_path
    except psutil.NoSuchProcess:
        return None

def get_system_uptime():
    """Function to get system uptime"""
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    return uptime

def get_boot_time():
    """Function returns system boot time"""
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
    return boot_time

def get_disk_io_stats():
    """Function to get disk IO stats"""
    return psutil.disk_io_counters(perdisk=False)

def main():
    """Main function does some system status check and calculations using numpy and pytorch"""
    print("System uptime: ", get_system_uptime())
    print("Boot Time: ", get_boot_time())
    print("Priority class for process id 1: ", get_priority_class(1))
    print("Memory info for process id 1: ", get_process_memory_info(1))
    print("Exe path for process id 1: ", get_process_exe_path(1))
    print("Disk I/O stats: ", get_disk_io_stats())

    tensor_a = torch.tensor([1.0, 2.0])
    tensor_b = torch.tensor([3.0, 4.0])
    result = tensor_a + tensor_b
    print("Result from Pytorch Lightning operation: ", result)

    np_array_a = np.array([1, 2, 3, 4])
    np_array_b = np.array([5, 6, 7, 8])
    result = np.add(np_array_a, np_array_b)
    print("Result from Numpy operation: ", result)


if __name__ == "__main__":
    main()

    print("\nTestcases scenario\n")

    with patch('psutil.boot_time', return_value=time.time()), \
            patch('psutil.Process', return_value=psutil.Process(5)):

        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print("Priority class for process id 5: ", get_priority_class(5))
        print("Memory info for process id 5: ", get_process_memory_info(5))
        print("Exe path for process id 5: ", get_process_exe_path(5))

    with patch('psutil.boot_time', return_value=time.time()-10000), \
            patch('psutil.Process', return_value=psutil.Process(3)):

        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print("Priority class for process id 3: ", get_priority_class(3))
        print("Memory info for process id 3: ", get_process_memory_info(3))
        print("Exe path for process id 3: ", get_process_exe_path(3))

    with patch('psutil.boot_time', return_value=time.time()), \
            patch('psutil.Process', side_effect=psutil.NoSuchProcess("No process found")):

        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print("Priority class for process id 1: ", get_priority_class(1))
        print("Memory info for process id 1: ", get_process_memory_info(1))
        print("Exe path for process id 1: ", get_process_exe_path(1))