import psutil
import datetime
import time
import os
import numpy as np
from unittest.mock import patch
import torch
from pytorch_lightning import seed_everything

# Seed value for Pytorch Lightning
seed_everything(42)

def get_processor_info():
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

def get_memory_info():
    print("Memory Information:")
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")

def get_disk_info():
    print("Disk Information:")
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f" Mountpoint: {partition.mountpoint}")
        print(f" File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        print(f" Total Size: {get_size(partition_usage.total)}")
        print(f" Used: {get_size(partition_usage.used)}")
        print(f" Free: {get_size(partition_usage.free)}")
        print(f" Percentage: {partition_usage.percent}%")

def get_network_info():
    print("Network Information:")
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            print(f"=== Interface: {interface_name} ===")
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                print(f"  MAC Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast MAC: {address.broadcast}")

# helper function to convert large number of bytes into a friendly format e.g 'KB', 'MB', 'GB', 'TB', 'PB'
def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def main():
    # print system related stats
    print("==="*10, "System Related Stats", "==="*10)
    print("System Boot Time:", get_boot_time())
    print("System Uptime: ", get_system_uptime())
    
    # print processor related stats
    print("==="*10, "Processor Related Stats", "==="*10)
    get_processor_info()
    
    # print memory related stats
    print("==="*10, "Memory Related Stats", "==="*10)
    get_memory_info()
    
    # print disk related stats
    print("==="*10, "Disk Related Stats", "==="*10)
    get_disk_info()
    
    # print network related stats
    print("==="*10, "Network Related Stats", "==="*10)
    get_network_info()    

if __name__ == "__main__":
    main()

    print("\nTestcases scenario\n")

    with patch('psutil.cpu_count', return_value=4):
        with patch('psutil.cpu_freq', return_value={'current': 2400.2, 'min': 1600.5, 'max': 3400.5}):
            get_processor_info()

    with patch('psutil.virtual_memory', return_value={'total': 8399302656, 'available': 4187330560, 'percent': 49.8, 'used': 4214667264}):
        get_memory_info()

    with patch('psutil.disk_partitions', return_value=[psutil._common.sdiskpart(device='C:\\', mountpoint='C:\\', fstype='NTFS', opts='rw,fixed')]):
        with patch('psutil.disk_usage', return_value={'total': 966571888640, 'used': 570699710464, 'free': 395872178176, 'percent': 45.8}):
            get_disk_info()
           
                
    with patch('psutil.net_if_addrs', return_value={'eth0': [psutil._common.snic(family=<AddressFamily.AF_INET: 2>, address='172.31.30.152', netmask='255.255.252.0', broadcast='172.31.31.255', ptp=None), psutil._common.snic(family=<AddressFamily.AF_PACKET: 17>, address='00:0c:29:68:22:4c', netmask=None, broadcast='ff:ff:ff:ff:ff:ff', ptp=None)]}):
        get_network_info()