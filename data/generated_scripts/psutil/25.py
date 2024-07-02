import time
import datetime
from unittest.mock import Mock
from unittest.mock import patch
import psutil

def get_process_info(process_id):
    process = psutil.Process(process_id)

    process_details = {
        'id': process.pid,
        'name': process.name(),
        'status': process.status(),
        'cpu_times': process.cpu_times(),
        'create_time': datetime.datetime.fromtimestamp(process.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
        'memory_info': process.memory_info(),
        'num_threads': process.num_threads(),
    }
    return process_details

def print_process_info(details, indent=0):
    lead_space = " "* indent
    print(f"{lead_space} Process information --- ")
    for key, value in details.items():
        if isinstance(value, dict):
            print(f"{lead_space} {key}:")
            print_process_info(value, indent+2)
        else:
            print(f"{lead_space} {key}: {value}")

def monitor_resources_for_60_seconds(process_id):
    print("Starting a 60 seconds monitor on process ", process_id)
    for i in range(60):
        p = psutil.Process(process_id)

        # Get CPU utilization
        cpu_percent = p.cpu_percent(interval=1)

        # Get memory utilization
        memory_info = p.memory_info()

        print(f"\tAt second {i+1}: CPU usage is {cpu_percent}%")
        print(f"\tAt second {i+1}: Memory usage is {memory_info.rss / (1024 * 1024)} MB")

        # Sleep for 1 second to get fresh data
        time.sleep(1)

def get_boot_time():
    boot_time_timestamp = psutil.boot_time()
    boot_time = datetime.datetime.fromtimestamp(boot_time_timestamp)
    return boot_time.strftime("%Y-%m-%d %H:%M:%S")
  
def print_system_info():
    ram_info = psutil.virtual_memory()
    print("System RAM Information: ", ram_info)

def main():
    # Print boot time 
    print('Boot time: ', get_boot_time())
  
    # Print memory details
    print_system_info()

    # Get the process details of the current Python script's process
    details = get_process_info(None)
    print_process_info(details)
    monitor_resources_for_60_seconds(details.get("id"))

if __name__ == "__main__":
    with patch('psutil.Process') as mock_p:
        mock_p.return_value = Mock(cpu_percent=Mock(return_value=10),
                                   memory_info=Mock(return_value=psutil._common.svmem(total=8376205312, available=1230237696, percent=85.3, used=6955038720, free=272758784, active=6788669440, inactive=752074752, buffers=367288320, cached=1774710784, shared=677535744, slab=187183104)),
                                   pid = id(mock_p))
        main()