import os
import time
import numpy as np
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process as Thread, current_process
from unittest.mock import Mock, patch
from random import randint


class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

def numpy_operation():
    # Sample array for numpy
    data = np.array([1, 2, 3])
    
    # Let's do a numpy operation here while the file is locked
    # This operation could be any critical section of your code
    square = np.square(data)
    print(f"Square of array {data} : {square}")
    return square

def pytorch_model():
    # PyTorch Lightning model
    model = LitModel()
    output = model.forward('sample_input')
    print(f"Model output for input 'sample_input': {output}")
    return output

def sleep_func():
    # Sleep for a while
    print(f"{current_process().name} is sleeping...")
    time.sleep(randint(1,3))
    print(f"{current_process().name} has woken up...")

def attempt_lock(lock_file):
    
    try:
        with FileLock(lock_file, timeout=1):
            print(f"{current_process().name} has the lock.")

            numpy_operation()
            pytorch_model()
            
            sleep_func()

            print(f"{current_process().name} has released the lock.")
            
            return True
            
    except Timeout:
        print(f"{current_process().name} failed to acquire the lock due to timeout, will retry after a while...")
        return False

def process_lock(lock_file):

    # Establish a random delay between lock attempts
    delay_between_attempts = randint(1, 3)
    
    print(f"{current_process().name} is attempting to acquire the lock...")

    has_lock = False

    while not has_lock:
        has_lock = attempt_lock(lock_file)
        
        if not has_lock:
            time.sleep(delay_between_attempts)

def make_processes(lock_file, n_processes):
    
    # Create a bunch of processes that will attempt to acquire the lock
    processes = [Thread(target=process_lock, args=(lock_file,)) for _ in range(n_processes)] 

    return processes

def main():
    try:
        # Define lock file path
        lock_path = "/tmp/my_lock_file"

        processes = make_processes(lock_path, 5)

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