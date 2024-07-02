import psutil
import time
from unittest.mock import patch
import datetime

# Get the system uptime
def get_system_uptime():
    """
    This function is used to get the system's uptime.
    Returns the system's uptime in the format of days, hours, minutes and seconds.
    """
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    return str(uptime).split('.')[0]  # Removing the microseconds part 


# Get the system boot time
def get_boot_time():
    """
    This function is used to get the system's boot time. The boot time here is the time
    since the system has been up and running.
    Returns the boot time in 'yyyy-mm-dd HH:MM:SS' format.
    """
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
    return boot_time.strftime('%Y-%m-%d %H:%M:%S')


# Get the process threads
def get_process_threads(pid):
    """
    This function is used to get the threads used by a particular process in the system.
    pid: process id for which the info has to be retrieved.
    Returns None if process doesn't exist, else returns the threads details.
    """
    try:
        process = psutil.Process(pid)
        return process.threads()
    except psutil.NoSuchProcess:
        return None


def main():
    """
    This function is used to print the system's uptime and boot time
    and the threads used by a given process id.
    """
    pid = int(input('Enter a process id: '))
    print("System uptime: ", get_system_uptime())
    print("Boot Time: ", get_boot_time())
    print(f"Threads for process id {pid}: ", get_process_threads(pid))


if __name__ == "__main__":
    main()
    print("\nTestcases scenario\n")

    # mock used for unit testing the methods
    pid = 1  # process id used for testcases

    # Testcase 1: Normal scenario where system is running and process id 1 is available
    with patch('psutil.boot_time', return_value=time.time()), \
            patch('psutil.Process', return_value=psutil.Process(pid)):
        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print(f"Threads for process id {pid}: ", get_process_threads(pid))

    # Testcase 2: Scenario where boot time was 10000 seconds before
    with patch('psutil.boot_time', return_value=time.time()-10000), \
            patch('psutil.Process', return_value=psutil.Process(pid)):
        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print(f"Threads for process id {pid}: ", get_process_threads(pid))

    # Testcase 3: Scenario where process is not available
    with patch('psutil.boot_time', return_value=time.time()), \
            patch('psutil.Process', side_effect=psutil.NoSuchProcess("No process found")):
        print("System uptime: ", get_system_uptime())
        print("Boot Time: ", get_boot_time())
        print(f"Threads for process id {pid}: ", get_process_threads(pid))