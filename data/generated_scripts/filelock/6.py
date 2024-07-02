import os
import time
import numpy as np
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process as Thread, Lock, current_process
from unittest.mock import Mock, patch
from random import randint

# Path of the lock file
lock_path = "/tmp/my_lock_file"

# Initializing the array for numpy
data = np.array([1, 2, 3])

# PyTorch Lightning model
class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

# Function handling the file lock
def process_lock(lock_file):

    delay_between_attempts = randint(1, 10)

    try:
        print(f"{current_process().name} is trying to acquire the lock...")
        
        while True:
            try:
                with FileLock(lock_file, timeout=1):
                    print(f"{current_process().name} got the lock.")
                    
                    # Perform tasks while having the lock

                    # Using numpy operation
                    square = np.square(data)
                    print(f"Numpy operation result : {square}")

                    # Using PyTorch Lightning model
                    model = LitModel()
                    print(f"Model Output : {model.forward('Hello')}")
                    print(f"Sleeping for a while...")
                    time.sleep(3)

                print(f"{current_process().name} has released the lock.")
                break

            except Timeout:
                print(f"{current_process().name} couldn't acquire the lock due to timeout! Retrying after delay..")
                time.sleep(delay_between_attempts)
    except Exception as e:
        print(f"{current_process().name} has encountered an unexpected error: {e}")


def another_process(lock_file):

    delay_between_attempts = randint(1, 10)

    try:
        print(f"{current_process().name} is trying to acquire the lock...")
        
        while True:
            try:
                with FileLock(lock_file, timeout=1):
                    print(f"{current_process().name} got the lock.")
                    
                    # Perform tasks while having the lock

                    square = np.square(data)
                    print(f"Numpy operation result : {square}")

                    model = LitModel()
                    print(f"Model Output : {model.forward('Hello')}")
                    print(f"Sleeping for a while...")
                    time.sleep(3)

                print(f"{current_process().name} has released the lock.")
                break

            except Timeout:
                print(f"{current_process().name} couldn't acquire the lock due to timeout! Retrying after delay..")
                time.sleep(delay_between_attempts)

    except Exception as e:
        print(f"{current_process().name} has encountered an unexpected error: {e}")


def main():
    try:
        # Create processes that will attempt to acquire the lock
        processes = []
        for _ in range(5):
            if _ % 2 == 0:
                processes.append(Thread(target=process_lock, args=(lock_path,)))
            else:
                processes.append(Thread(target=another_process, args=(lock_path,))) 

        # Start all processes
        for process in processes:
            process.start()

        # Wait for all processes to finish their tasks
        for process in processes:
            process.join()

    except Exception as e:
        print(f"There has been an error in the main function: {str(e)}")


if __name__ == "__main__":
    main()