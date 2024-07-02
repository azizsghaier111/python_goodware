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
        return x

# Function to mock heavy computational task
def heavy_computation(data):
    print(f"{current_process().name}: Starting heavy computation.")
    time.sleep(10)
    return np.square(data)

# File lock function
def file_lock(lock_path, guess_count=5):
    attempt = 0
    print(f"{current_process().name}: Attempting to acquire the lock.")
    while attempt < guess_count:
        try:
            start_time = time.time()
            with FileLock(lock_path, timeout=1):
                print(f"{current_process().name}: Lock {lock_path} is acquired.")
                heavy_computation(data)
                print(f"{current_process().name}: Lock {lock_path} is released.")
            time_taken = time.time() - start_time
        except Timeout:
            print(f"{current_process().name}: Failed to acquire lock {lock_path}")
            attempt += 1
        else:
            break

        if attempt > 0:
            print(f"{current_process().name} has attempt {attempt} times.")

        print(f"{current_process().name}: Time taken is {time_taken}.")

        if time_taken > 1000:
            print(f"{current_process().name}: Time taken has crossed 1000.")

    else:
        print(f"{current_process().name}: Time taken is None.")

if __name__ == "__main__":
    p = process_lock(LOCK_PATH)
    p.start()
    main()