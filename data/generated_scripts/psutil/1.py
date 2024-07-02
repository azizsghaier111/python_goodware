import psutil           # Provides an interface to get information on running processes
import datetime         # Module to manipulate dates and times
import time             # Module provides various functions to manipulate time values
from unittest.mock import patch    # Mock class for testing in Python

# System uptime functionality
def get_system_uptime():
    """
    This function returns the system's uptime.
    :return: string: System's uptime
    """
    # Current time - System boot time
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    return uptime

# System boot time functionality
def get_boot_time():
    """
    This function returns the system boot time.
    :return: string: System boot time
    """
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
    return boot_time

# Process-related functionalities
def get_process_threads(process_id):
    """
    This function returns the threads associated with a given process.
    :param process_id: ID of the process to get its threads
    :return: list: List of threads for the process
    """
    try:
        # Get the process by its id
        process = psutil.Process(process_id)
        # Get the threads of the process
        return process.threads()
    except psutil.NoSuchProcess:
        # Return None if the process does not exist
        return None

def main():
    """
    Main function to print system uptime, boot time and threads for a process.
    """
    print("System uptime: ", get_system_uptime())
    print("Boot Time: ", get_boot_time())
    print("Threads for process id 1: ", get_process_threads(1))

if __name__ == "__main__":
    # Call the main function
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