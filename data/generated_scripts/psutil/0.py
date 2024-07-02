import datetime
import psutil
import time
from unittest.mock import patch


def get_system_uptime():
    # Get the system uptime
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    return uptime


def get_boot_time():
    # Get the system boot time
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
    return boot_time


def get_process_threads(process_id):
    # Get the process threads
    try:
        process = psutil.Process(process_id)
        return process.threads()
    except psutil.NoSuchProcess:
        return None


def main():
    print("System uptime: ", get_system_uptime())
    print("Boot Time: ", get_boot_time())
    print("Threads for process id 1: ", get_process_threads(1))


if __name__ == "__main__":
    main()
    print("\nTestcases scenario\n")

    # Testcases scenarios using mock
    with patch('psutil.boot_time', return_value=time.time()), \
            patch('psutil.Process', return_value=psutil.Process(1)):

        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print("Threads for process id 1: ", get_process_threads(1))

    with patch('psutil.boot_time', return_value=time.time()-10000), \
            patch('psutil.Process', return_value=psutil.Process(1)):

        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print("Threads for process id 1: ", get_process_threads(1))

    with patch('psutil.boot_time', return_value=time.time()), \
            patch('psutil.Process', side_effect=psutil.NoSuchProcess("No process found")):

        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print("Threads for process id 1: ", get_process_threads(1))