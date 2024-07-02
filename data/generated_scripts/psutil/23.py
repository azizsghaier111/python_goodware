import psutil
import datetime


# Additional libraries for User-related functionalities
import os
import getpass


# Additional library for process priority functionality
import subprocess


# Process Information Function
def get_process_info(process_id):
    # Checks if the process_id is None and uses the current process id
    if process_id is None:
        process_id = os.getpid()

    # Attempt to access process info
    try:
        p = psutil.Process(process_id)

        # Get process details
        details = {
            "name": p.name(),
            "exe": p.exe(),
            "cwd": p.cwd(),
            "status": p.status(),
            "create_time": datetime.datetime.fromtimestamp(p.create_time()),
            "uids": p.uids(),
            "gids": p.gids(),
            "cpu_times": p.cpu_times(),
            "memory_info": p.memory_info(),
            "io_counters": p.io_counters(),
            "open_files": p.open_files(),
            "num_fds": p.num_fds() if hasattr(p, "num_fds") else None,
            "num_threads": p.num_threads(),
            "threads": p.threads(),
            "children": [child.as_dict(attrs=['pid', 'name', 'status']) for child in p.children()],
            "parent": p.parent().as_dict(attrs=['pid', 'name', 'status']) if p.parent() else None,
            "connections": p.connections()
        }

        return details

    except psutil.NoSuchProcess:
        print(f"No such process: {process_id}")
        return


# Disk Usage Information
def get_disk_usage():
    print("\n*** Disk Usage ***")
    for disk in psutil.disk_partitions(all=False):
        usage = psutil.disk_usage(disk.mountpoint)
        print(f"Partition {disk.device} - Total: {usage.total // (2**30)} GiB, Used: {usage.used // (2**30)} GiB, Free: {usage.free // (2**30)} GiB")


# CPU Information
def get_cpu_info():
    print("\n*** CPU Info ***")
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    print(f"Max Frequency: {psutil.cpu_freq().max:.2f}Mhz")
    print(f"Min Frequency: {psutil.cpu_freq().min:.2f}Mhz")
    print(f"Current Frequency: {psutil.cpu_freq().current:.2f}Mhz")
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")


# Memory Information
def get_memory_info():
    print("\n*** Memory Information ***")
    svmem = psutil.virtual_memory()
    print(f"Total: {svmem.total // (2**30)}GB")
    print(f"Available: {svmem.available // (2**30)}GB")
    print(f"Used: {svmem.used // (2**30)}GB")
    print(f"Percentage: {svmem.percent}%")


# Network Information
def get_network_info():
    print("\n*** Network Information ***")
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"*** {interface_name} ***")
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")


# User-related functionalities
def get_user_info():
    print("\n*** User Information ***")
    print(f'Logged user: {getpass.getuser()}')
    print(f'Home: {os.path.expanduser("~")}')


# Process priority related functionalities
def execute_shell_cmds():
    print("\n*** Process Priority ***")
    # set nice value to a lower priority
    subprocess.Popen("ps", shell=True, executable="/bin/bash",
                     env=dict(os.environ, SCHEDULER="batch"))
    # observe the result
    subprocess.Popen("ps -l", shell=True, executable="/bin/bash")


def main():
    # Get the process details of the current Python script's process
    process_info = get_process_info(None)
    print(process_info)

    # Get disk usage
    get_disk_usage()

    # Get CPU info
    get_cpu_info()

    # Get Memory info
    get_memory_info()

    # Get Network info
    get_network_info()

    # Get User info
    get_user_info()

    # Execute shell commands with process priority
    execute_shell_cmds()


if __name__ == "__main__":
    main()