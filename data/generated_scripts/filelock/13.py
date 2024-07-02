import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process, current_process
from unittest.mock import Mock

print("Launching Python script.\n")

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


def process_lock(lock_path, retry_count=5):

    lock = FileLock(lock_path)
    attempt = 0

    while True:
        try:
            with lock.acquire(timeout=10):
                print(f"Process {current_process().name} has acquired the lock.")
                # heavy computation
                print(f"Process {current_process().name} is performing heavy computation.")
                result = heavy_computation(data)
                print(f"Result of heavy computation is: {result}")
                model_output = LitModel().forward('Hello')
                print(f"Model output: {model_output}")

                # simulate workload
                print(f"Process {current_process().name} is simulating workload.")
                time.sleep(np.random.randint(1, 5))
                print('Simulation complete.')
                break
        except Timeout:
            # handling timeout
            attempt += 1
            if attempt < retry_count:
                print(f"Process {current_process().name} couldn't acquire lock. Retrying...")
                continue
            else:
                print(f"Maximum retry limit reached for process {current_process().name}. Skipping...")
                break
        except Exception as e:
           # handling other exceptions
            print(f"Error encountered: {e}")
            break


def main():
    try:
        # creating lock instances
        processes = [Process(target=process_lock, args=(lock_path,)) for _ in range(5)]
        
        # start all processes
        for process in processes:
            process.start()
        
        # wait for all processes to finish
        for process in processes:
            process.join()
    except Exception as e:
        print(f"The main function failed due to the following error: {str(e)}")


print('Initiation...\n')
if __name__ == "__main__":
    main()
print('\nEnd of the script.')