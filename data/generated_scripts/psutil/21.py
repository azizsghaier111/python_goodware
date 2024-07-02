import os
import sys
import datetime
import psutil 

def get_system_memory():
    """
    Function to get system memory details.
    """
    mem = psutil.virtual_memory()
    details = {
        "total": mem.total,
        "available": mem.available,
        "percent": mem.percent,
        "used": mem.used,
        "free": mem.free,
        "active": mem.active,
        "inactive": mem.inactive,
        "buffers": mem.buffers,
        "cached": mem.cached,
        "shared": mem.shared,
        "slab": mem.slab
    }
    return details


def get_system_disk_usage():
    """
    Function to get system disk usage details.
    """
    disk = psutil.disk_usage('/')
    details = {
        "total": disk.total,
        "used": disk.used,
        "free": disk.free,
        "percent": disk.percent
    }
    return details


def get_system_info():
    """
    Function to get system-wide information.
    """
    details = {
        "cpu_count": psutil.cpu_count(),
        "cpu_times": psutil.cpu_times(),
        "boot_time": datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        "memory": get_system_memory(),
        "disk_usage": get_system_disk_usage()
    }
    return details


def print_system_info(details):
    """
    Function to print system-wide information.
    """
    print("CPU count:", details["cpu_count"])
    print("CPU times:", details["cpu_times"])
    print("Boot time:", details["boot_time"])
    print("Memory details:")
    for k, v in details["memory"].items():
        print("\t", k, ":", v)
    print("Disk usage details:")
    for k, v in details["disk_usage"].items():
        print("\t", k, ":", v)


def main():
    """
    Main function to print the process and system info.
    """
    pid = os.getpid()
    print("\nCurrent Process ID:", pid)
    print("Current Process Info:")
    print_process_info(get_process_info(pid))
    print("\nSystem Info:")
    print_system_info(get_system_info())

if __name__ == "__main__":
    main()