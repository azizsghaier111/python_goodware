import os
import filelock
import time
import pytest
from unittest import mock
import pytorch_lightning as pl
import numpy as np
from threading import Thread

# The path to the lock file
lock_path = "/tmp/my_lock_file"

# Sample array for numpy
data = np.array([1, 2, 3])

# Sample Model for PyTorch Lightning
class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

def task_with_lock():
    # Exception handling during file lock
    try:
        # filelock is thread safe and compatible with Context Management Protocol 
        with filelock.FileLock(lock_path, timeout=1):
            # using numpy operation
            square = np.square(data)
            print(square)

            # PyTorch Lightning Model
            model = LitModel()
            print(model.forward('Hello'))

            # some I/O operation
            time.sleep(3)
    except filelock.Timeout:
        print("Another instance is running.")
    except FileNotFoundError:
        print("Lock file not found.")

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

# Main function
def main():

    # Initialize Threads
    threads = []

    # Create and start a new thread for each task
    for _ in range(5):
        thread = Thread(target=task_with_lock)

        threads.append(thread)

        thread.start() 

    # Wait for all threads to complete
    for thread in threads: 
        thread.join() 

if __name__ == "__main__":
    main()