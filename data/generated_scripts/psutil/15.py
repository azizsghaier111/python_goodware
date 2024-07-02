import psutil
import datetime
import time


def get_process_info(process_id):
    # Attempt to access process info
    try:
        p = psutil.Process(process_id)

        # Get process details
        details = {"name": p.name(),
                   "exe": p.exe(),
                   "cwd": p.cwd(),
                   "status": p.status(),
                   "create_time": datetime.datetime.fromtimestamp(p.create_time()).isoformat(),
                   "uids": p.uids(),
                   "gids": p.gids(),
                   "cpu_times": p.cpu_times(),
                   "memory_info": p.memory_info(),
                   "io_counters": p.io_counters(),
                   "open_files": p.open_files(),
                   "num_fds": p.num_fds() if hasattr(p, "num_fds") else None,
                   "num_threads": p.num_threads(),
                   "threads": p.threads(),
                   "children": [get_process_info(child.pid) for child in p.children()],
                   "parent": get_process_info(p.ppid()) if p.ppid() else None,
                   "connections": p.connections()
                   }

    except psutil.NoSuchProcess:
        details = None
    return details


def get_disk_usage():
    print("\n### Disk Usage ###")
    for part in psutil.disk_partitions(all=False):
        if part.fstype != '':
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except PermissionError:
                continue
            else:
                print("Mountpoint:", part.mountpoint)
                print("Total:", usage.total)
                print("Used:", usage.used)
                print("Free:", usage.free)
                print("Percentage:", usage.percent, " %\n")


def get_cpu_info():
    print("\n### CPU Info ###")
    print("CPU Count (logical):", psutil.cpu_count(logical=True))
    print("CPU Count (physical):", psutil.cpu_count(logical=False))
    print("CPU Times:", psutil.cpu_times())
    print("CPU Stats:", psutil.cpu_stats())
    print("CPU Utilization (per CPU):", psutil.cpu_percent(percpu=True), "\n")


def get_sys_mem_info():
    print("\n### System Memory Info ###")
    mem = psutil.virtual_memory()
    print('Total:', mem.total)
    print('Available:', mem.available)
    print('Used:', mem.used)
    print('Percent:', mem.percent)
    print('Active:', mem.active)
    print('Inactive:', mem.inactive)
    print('Buffers:', mem.buffers)


def get_swap_mem_info():
    print("\n### Swap Memory Info ###")
    swap = psutil.swap_memory()
    print('Swap Total:', swap.total)
    print('Swap Used:', swap.used)
    print('Swap Free:', swap.free)
    print('Swap Percent:', swap.percent)


def get_net_info():
    print("\n### Network Info ###")
    net_io = psutil.net_io_counters()
    print('Bytes Sent:', net_io.bytes_sent)
    print('Bytes Received:', net_io.bytes_recv)
    print('Packets Sent:', net_io.packets_sent)
    print('Packets Received:', net_io.packets_recv)
    print('Err Input:', net_io.errin)
    print('Err Output:', net_io.errout)


def main():
    # Get the process details of the current Python script's process
    # print_process_info(get_process_info(None))

    # Get disk usage
    get_disk_usage()

    # Get CPU info
    get_cpu_info()

    # Get System Memory info
    get_sys_mem_info()

    # Get Swap Memory info
    get_swap_mem_info()

    # Get Network info
    get_net_info()


if __name__ == "__main__":
    main()