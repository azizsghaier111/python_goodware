import os
import filelock
import time
import numpy as np
import pytorch_lightning as pl
from unittest.mock import patch
from multiprocessing import Process as Thread, Lock, current_process
from filelock import FileLock, Timeout

# Define lock file path
lock_path = "/tmp/my_lock_file"

# Sample array for numpy
data = np.array([1, 2, 3])

# Sample Model for PyTorch Lightning
class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

# Process lock function
def process_lock(lock_file):

    try:
        print(f"{current_process().name} is attempting to acquire the lock...")
        with FileLock(lock_file):
            # using numpy operation
            square = np.square(data)
            print(f"{current_process().name} has the lock. Square: {square}")

            # PyTorch Lightning Model
            model = LitModel()
            print(f"{current_process().name} Model output: {model.forward('Hello')}")

            # some I/O operation
            print(f"{current_process().name} Sleeping...")
            time.sleep(3)
        print(f"{current_process().name} has released the lock")

    except Timeout:
        print(f"{current_process().name} failed to acquire the lock")

    except Exception as e:
        print(f"{current_process().name} Unexpected error: {str(e)}")

def main():

    try:
        # Lock object to ensure only one process is writing to the file at a time
        lock = Lock()

        # Create a new process for each task
        processes = [Thread(target=process_lock, args=(lock_path,))
                     for _ in range(5)]

        # Start the processes
        for process in processes:
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")

if __name__ == "__main__":
    main()