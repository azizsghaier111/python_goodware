import datetime
import time
from unittest.mock import patch
import psutil


def get_process_info(process_id):
    # ...
    # The same code as before
    # ...


def print_process_info(details, indent=0):
    # ...
    # The same code as before
    # ...


def monitor_resources_for_60_seconds(process_id):
    print("Starting a 60 seconds monitor on process ", process_id)
    for i in range(60):
        p = psutil.Process(process_id)

        # Get CPU utilization
        cpu_percent = p.cpu_percent(interval=1)

        # Get memory utilization
        memory_info = p.memory_info()

        print(f"At second {i+1}: CPU usage is {cpu_percent}%")
        print(f"At second {i+1}: Memory usage is {memory_info.rss / (1024 * 1024)} MB")
        time.sleep(1)


def main():
    # Test the process details of the current Python script's process
    details = get_process_info(None)
    print_process_info(details)
    monitor_resources_for_60_seconds(details.get("id"))


if __name__ == "__main__":
    with patch('psutil.Process') as mock_p:
        mock_p.return_value.pid = id(mock_p)
        main()