import os
import time
import numpy as np
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process as Thread, Lock, current_process
from unittest.mock import Mock, patch
from random import randint


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

    delay_between_attempts = randint(1, 10)

    try:
        print(f"{current_process().name} is attempting to acquire the lock...")
        
        while True:
            try:
                with filelock.FileLock(lock_file, timeout=1):
                    print(f"{current_process().name} has the lock.")
                    
                    # Do some work here
                    # using numpy operation
                    square = np.square(data)
                    print(f"Square: {square}")

                    # PyTorch Lightning model
                    model = LitModel()
                    print(f"Model output: {model.forward('Hello')}")

                    print(f"{current_process().name} is sleeping...")
                    time.sleep(3)
                
                print(f"{current_process().name} has released the lock.")
                break

            except Timeout:
                print(f"{current_process().name} failed to acquire the lock due to timeout, retrying after delay...")
                time.sleep(delay_between_attempts)

    except Exception as e:
        print(f"{current_process().name} experienced an unexpected error: {e}")

def main():
    try:
        # Create a bunch of processes that will attempt to acquire the lock
        processes = [Thread(target=process_lock, args=(lock_path,)) for _ in range(5)] 

        # Start each process
        for process in processes:
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

    except Exception as e:
        print(f"An error occurred in main function: {str(e)}")

if __name__ == "__main__":
    main()