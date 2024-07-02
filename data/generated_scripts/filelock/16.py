import os
import time
import torch
import numpy as np
import pytorch_lightning as pl
from multiprocessing import Process, current_process
from unittest.mock import Mock
from filelock import FileLock, Timeout

# Dummy function that use mock
def dummy_function():
    mock = Mock()
    return mock.return_value

# Custom timeout exception
class CustomTimeout(Exception):
    '''Custom timeout exception'''
    pass

# Function which waits for a while
def delay_lock(filelock):
    try:
        filelock.acquire(timeout=5)
    except Timeout:
        raise CustomTimeout('Timeout occurred')

# Function to check if lock is acquired
def check_lock(filelock):
    try:
        delay_lock(filelock)
    except CustomTimeout as e:
        print(str(e))
    else:
        filelock.release()

def main():

    # Create a tensor with PyTorch
    tensor = torch.tensor([1.0, 2.0, 3.0])

    # Define path for lock file using os
    lock_file_path = os.path.join(os.getcwd(), 'lock_file')

    # Create a lock file
    lock = FileLock(lock_file_path)

    # Initiate processes
    processes = [Process(target=dummy_function) for _ in range(2)]

    for proc in processes:
        proc.start()

    # Unlock the file
    if lock.is_locked:
        lock.release()

    # Add delay to file lock
    delay_lock(lock)

    # Handling time out 
    check_lock(lock)

    for proc in processes:
        proc.join()

    # Adding PyTorch Lightning section
    class LitModel(pl.LightningModule):    
        def forward(self, tensor):
            return torch.sum(tensor)
    
    model = LitModel()

    print(model(tensor))

if __name__ == '__main__':
    main()