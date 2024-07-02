import os
import filelock
import time
import numpy as np
import pytorch_lightning as pl
from unittest.mock import patch
from multiprocessing import Process as Thread, Lock, current_process
from filelock import Timeout

# Define lock file path
lock_path = "/tmp/my_lock_file"

# Sample array for numpy
data = np.array([1, 2, 3])

# Sample Model for PyTorch Lightning
class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

# Numpy demonstration
def numpy_demo():
    print(f"{current_process().name} Performing numpy operations...")
    square = np.square(data)
    print(f"Square of data: {square}")
    sum_of_squares = np.sum(square)
    print(f"Sum of squares: {sum_of_squares}")
    
    return sum_of_squares

# PyTorch Lightning demonstration
def pytorch_demo():
    print(f"{current_process().name} Testing PyTorch Lightning model...")
    model = LitModel()
    model_output = model.forward('Hello World')
    print(f"Model output: {model_output}")

    return model_output

# Process lock function
def process_lock(lock_file):
    try:
        print(f"{current_process().name} is attempting to acquire the lock...")
        with FileLock(lock_file):
            print(f"{current_process().name} has the lock.")

            numpy_demo()
            pytorch_demo()
            
            # Introducing some I/O bound delay
            print(f"{current_process().name} Sleeps for a while...")
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
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()