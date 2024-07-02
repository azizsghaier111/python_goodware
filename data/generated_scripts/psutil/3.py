import datetime
import time
from unittest.mock import patch
import psutil


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
                   "connections": p.connections()}

    except psutil.NoSuchProcess:
        details = None

    return details


def print_process_info(details, indent=0):
    if details is None:
        return
    print(" " * indent, "PID:", details["name"])
    print(" " * indent, "EXE:", details["exe"])
    print(" " * indent, "CWD:", details["cwd"])
    print(" " * indent, "Status:", details["status"])
    print(" " * indent, "Created:", details["create_time"])
    print(" " * indent, "UIDs:", details["uids"])
    print(" " * indent, "GIDs:", details["gids"])
    print(" " * indent, "CPU Times:", details["cpu_times"])
    print(" " * indent, "Memory Info:", details["memory_info"])
    print(" " * indent, "IO Counters:", details["io_counters"])
    print(" " * indent, "Open Files:", details["open_files"])
    print(" " * indent, "Num FDs:", details["num_fds"])
    print(" " * indent, "Num Threads:", details["num_threads"])
    print(" " * indent, "Threads:", details["threads"])

    for child in details["children"]:
        print(" " * indent, "# Child Process")
        print_process_info(child, indent + 2)

    print(" " * indent, "# Parent Process")
    print_process_info(details["parent"], indent + 2)
    print(" " * indent, "Connections:", details["connections"])


def main():
    # Test the process details of the current Python script's process
    print_process_info(get_process_info(None))


if __name__ == "__main__":
    main()