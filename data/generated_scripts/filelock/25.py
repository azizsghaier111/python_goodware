import os
import filelock
import time
import pytest
from unittest import mock
import pytorch_lightning as pl
import numpy as np
from random import randint

# The path to the lock file
lock_path = "/tmp/my_lock_file"

# Sample array for numpy
data = np.array([1, 2, 3])

# Sample Model for PyTorch Lightning
class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

def retry_on_failure(delay=1, max_retries=5):
    # Function wrapper for retry functionality
    def decorator_retry(func):
        def wrapper_retry(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except filelock.Timeout:
                    print(f'Failed to acquire lock, retrying in {delay} seconds')
                    time.sleep(delay)
                    continue
            return func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry

@retry_on_failure(delay=randint(1, 10), max_retries=9)
def task_with_lock():
    # Just some task that can be run in different processes
    try:
        # Set a delay for other parallel tasks
        lock = filelock.FileLock(lock_path, timeout=1)

        with lock:
            # Customizable random delay
            delay = randint(1, 5)
            print(f'Got lock after {delay}s delay')
            time.sleep(delay)

            # Using numpy operation
            square = np.square(data)
            print(square)

            # PyTorch Lightning Model
            model = LitModel()
            print(model.forward('Hello'))

            # Some I/O operation
            time.sleep(1)

            print("Task complete")
    except filelock.Timeout:
        print("Couldn't get lock")
        return

# Test script to validate the task with Mocks
@pytest.fixture
def mock_filelock(monkeypatch):
    m = mock.Mock()
    monkeypatch.setattr(filelock, 'FileLock', m)
    return m

def test_given_lockfile_when_multipleInstances_then_raise_interruptedError(mock_filelock):
    with pytest.raises(filelock.Timeout):
        task_with_lock()
        mock_filelock.assert_called_once_with(lock_path)
        
def main():
    # A list to hold the processes
    processes = []

    try:
        # Run 5 tasks in parallel
        for i in range(5):
            # Create a process and start it
            process = task_with_lock()
            processes.append(process)

        # Wait for all processes to finish
        for process in processes:
            process
    except filelock.Timeout:
        print("Another instance is running.")
    except FileNotFoundError:
        print("Lock file not found.")

if __name__ == "__main__":
    main()