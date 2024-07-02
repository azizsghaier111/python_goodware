import psutil
import datetime
import time
import os
import numpy as np
from unittest.mock import patch
import torch
from pytorch_lightning import seed_everything

# Seeding for reproducible results
seed_everything(42)

# psutil processes related functions
def get_process_memory_info(process_id):
    try:
        process = psutil.Process(process_id)
        exe_path = process.memory_info()
        return exe_path
    except psutil.NoSuchProcess:
        return None
def get_process_exe_path(process_id):
    try:
        process = psutil.Process(process_id)
        exe_path = process.exe()
        return exe_path
    except psutil.NoSuchProcess:
        return None

# Prints the system uptime
def print_system_uptime():
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
    sys_uptime = datetime.datetime.now() - boot_time
    print(f"System Uptime is: {sys_uptime}")
    return sys_uptime

# Prints the boot time
def print_boot_time():
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
    print(f"Boot Time is: {boot_time}")
    return boot_time

# Main function
def main():
    # Using the above defined functions
    print_system_uptime()
    print_boot_time()
    print("Memory info for process id 1: ", get_process_memory_info(1))
    print("Exe path for process id 1: ", get_process_exe_path(1))

    # Pytorch and numpy operations
    tensor_a = torch.tensor([1.0, 2.0])
    tensor_b = torch.tensor([3.0, 4.0])

    result = tensor_a + tensor_b
    print("Result from Pytorch operation: ", result)

    np_array_a = np.array([1, 2, 3, 4])
    np_array_b = np.array([5, 6, 7, 8])

    result = np.add(np_array_a, np_array_b)
    print("Result from Numpy operation: ", result)

if __name__ == "__main__":
    main()

    print("\nTestcases scenario\n")

    with patch('psutil.boot_time', return_value=time.time()), \
            patch('psutil.Process.memory_info', return_value=("[memrss=56789, mem_vms=1234567]")):

        print_system_uptime()
        print_boot_time()
        print("Memory info for process id 1: ", get_process_memory_info(1))
    with patch('psutil.Process.exe', return_value=("/usr/libexec/platform-python-3.8")):

        print("Exe path for process id 1: ", get_process_exe_path(1))