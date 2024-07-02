import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process, Lock, current_process
from unittest.mock import Mock

# Define lock file path
lock_path = "/tmp/my_lock_file"

# Sample array for numpy
data = np.array([1, 2, 3])

# Sample Model for PyTorch Lightning
class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

# Function to mock heavy computational task
def heavy_computation(data):
    time.sleep(10) # Mock long running task
    return np.square(data)

# Process lock function
def process_lock(lock_path, retry_count=5):

    lock = FileLock(lock_path)
    attempt = 0

    while attempt < retry_count:
        try:
            print(f"{current_process().name} is attempting to acquire the lock...")
            with lock.acquire(timeout=10):
                print(f"{current_process().name} has the lock.")

                # Do some work here
                result = heavy_computation(data)
                print(f"Result: {result}")

                # PyTorch Lightning model
                model = LitModel()
                print(f"Model output: {model.forward('Hello')}")

                # Simulating more workload
                time_taken = torch.Tensor([1.0])
                for _ in range(10**7):
                    time_taken = time_taken + 0.1
                    
                time.sleep(np.random.randint(1, 5))  # add variability

            print(f"{current_process().name} released the lock.")
            return

        except Timeout:
            attempt += 1
            print(f"{current_process().name} failed to acquire the lock. Retrying...")

        except Exception as e:
            print(f"{current_process().name} experienced an error: {e}")
            return

    print(f"{current_process().name} failed to acquire the lock after {retry_count} attempts.")
    return

def main():
    try:
        # Start a bunch of processes that will attempt to acquire the lock
        processes = [Process(target=process_lock, args=(lock_path,)) for _ in range(5)] 

        for process in processes:
            process.start()

        for process in processes:
            process.join()

    except Exception as e:
        print(f"Failed in main function: {str(e)}")

if __name__ == "__main__":
    main()