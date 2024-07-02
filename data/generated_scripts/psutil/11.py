import os
import getpass
import datetime
import time
import psutil
import numpy as np
from unittest.mock import patch

# Users-Related Functionalities
def active_users():
    return psutil.users()

def print_active_users(users):
    print("Active users:")
    for user in users:
        print(f"Name: {user.name}, Terminal: {user.terminal}, Host: {user.host}, Started: {datetime.datetime.fromtimestamp(user.started)}")

# Check If a Process is Running Functionalities
def check_process_running(process_name):
    for proc in psutil.process_iter(['name']):
        if process_name == proc.info['name']:
            return True
    return False

# Process-Related Functionalities
def get_process_info(process_id):
    try:
        p = psutil.Process(process_id)
        return p.as_dict(attrs=['pid', 'name', 'username'])

    except psutil.NoSuchProcess:
        return None

def print_process_info(proc_info):
    if proc_info:
        print(f"PID: {proc_info['pid']}, Name: {proc_info['name']}, Username: {proc_info['username']}")

def all_processes():
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        print_process_info(proc.info)

# System Information Reporting
def system_info():
    print(f"System CPU count: {psutil.cpu_count()}")
    print(f"System CPU times: {psutil.cpu_times()}")
    print(f"System memory usage: {psutil.virtual_memory()}")
    print(f"System disk usage: {psutil.disk_usage('/')}")
    print(f"System boot time: {datetime.datetime.fromtimestamp(psutil.boot_time())}")

# Execution Entry Point
def main():
    # Users-related functionalities
    print_active_users(active_users())

    # Check if a process is running functionalities
    process_name = 'python'
    print(f"Is {process_name} running? {check_process_running(process_name)}")

    # Process-related functionalities
    cur_process_info = get_process_info(os.getpid())
    print_process_info(cur_process_info)
    print("All processes:")
    all_processes()

    # System info reporting
    system_info()

if __name__ == "__main__":
    main()