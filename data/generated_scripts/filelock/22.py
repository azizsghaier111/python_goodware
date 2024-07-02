import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process, Lock, current_process
from unittest.mock import Mock

# Define lock file path
LOCK_PATH = "/tmp/my_lock_file"

# Sample array for numpy
data = np.array([1, 2, 3, 4])

# Sample Model for PyTorch Lightning
class LitModel(pl.LightningModule):
    def forward(self, x):
        return torch.relu(x)

# Function to mock heavy computational task
def heavy_computation(data, id):
    print(f"Process {id}: Starting heavy computation.")
    time.sleep(10)  # Simulate heavy computation
    return np.square(data)

# Function to delay to retry lock acquisition on failure
def delay_on_retry(attempts, max_delay=5):
    delay = min(2 ** attempts, max_delay)
    time.sleep(delay)

# File lock function with decorator
def with_filelock(lock_path):
    def decorator(function):
        def wrapper(*args, **kwargs):
            lock = FileLock(lock_path)
            with lock:
                return function(*args, **kwargs)
        return wrapper
    return decorator

@with_filelock(lock_path=LOCK_PATH)
def process_task(data, id):
    print(f"Process {id} has acquired the file lock. Performing heavy computation.")
    result = heavy_computation(data, id)
    print(f"Process {id} completed heavy computation. Result: {result}. Releasing file lock.")
    return result

if __name__ == "__main__":
    num_processes = 4  # Define the number of processes to run concurrently
    processes = []
    for i in range(num_processes):
        process = Process(target=process_task, args=(data, i))
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Join all processes
    for process in processes:
        process.join()