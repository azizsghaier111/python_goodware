import os
import time
import random
import numpy as np
import torch
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process, current_process
from unittest.mock import Mock


print("\n-------------- Launching Python script --------------\n")


process_names = [f"Process-{str(i)}" for i in range(5)]
lock_path = os.getcwd() + "/temp_lock_file"  # use the current directory for lock file creation
data = np.array([1, 2, 3])


class LitModel(pl.LightningModule):
    """ 
    This class defines a simple PyTorch Lightning module.
    """
    def forward(self, x):
        return x


def heavy_computation(data):
    time.sleep(10)
    result = np.square(data)
    for i in range(5):
        result += i  
        result -= i  
    return result


def get_mock_process():
    # Creating a mock process to simulate additional workload
    mock_process = Mock()
    mock_process.start = Mock(return_value=None)
    mock_process.join = Mock(return_value=None)
    return mock_process


def process_lock(lock_path, retry_count=5):

    lock = FileLock(lock_path)
    attempt = 0

    while True:
        try:
            with lock.acquire(timeout=10):
                print(f"\n{current_process().name} has acquired the lock.\n")
                # heavy computation
                print(f"{current_process().name} is performing heavy computation.\n")
                result = heavy_computation(data)
                print(f"Result of heavy computation for {current_process().name} is: {result}")

                mock_model_output = LitModel().forward('Hello, World')
                print(f"\nMock Model for {current_process().name}, output: {mock_model_output}\n")

                # simulate workload with mock process
                print(f"\n{current_process().name} is simulating workload with a mock process\n")
                mock_process = get_mock_process()
                mock_process.start()
                mock_process.join()
                time.sleep(random.randint(2, 10))  # adding randomness to simulate real-world scenarios better
                print('\nSimulation complete for the current process\n')
                break
        except Timeout:
            # handling timeout
            attempt += 1
            if attempt < retry_count:
                print(f"\n{current_process().name} couldn't acquire lock. Retrying after random delay.\n")
                time.sleep(random.randint(1, 5))  # random delay to simulate real-world scenarios better
                continue
            else:
                print(f"\nMaximum retry limit reached for {current_process().name}. Skipping...\n")
                break
        except Exception as e:
           # handling other exceptions
            print(f"\nError encountered: {e}\n")
            break


def main():
    try:
        # creating processes instances
        processes = [Process(target=process_lock, args=(lock_path,), name=process_name) for process_name in process_names]
        
        # start all processes
        for process in processes:
            process.start()
        
        # wait for all processes to finish
        for process in processes:
            process.join()
    except Exception as e:
        print(f"\nThe main function failed due to the following error: {str(e)}")


if __name__ == "__main__":
    print('\n-------------- Initiation --------------\n')
    main()
    print('\n-------------- End of the script --------------\n')