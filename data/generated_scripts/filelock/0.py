# Imports
import os
import filelock
import time
import pytest
from unittest import mock
import pytorch_lightning as pl
import numpy as np

# The path to the lock file
lock_path = "/tmp/my_lock_file"

# Sample array for numpy
data = np.array([1, 2, 3])

# Sample Model for PyTorch Lightning
class LitModel(pl.LightningModule):
    def forward(self, x):
        return x


def task_with_lock():
    # InterruptedError will be raised if the file is already locked
    with filelock.FileLock(lock_path):
        # using numpy operation
        square = np.square(data)
        print(square)

        # PyTorch Lightning Model
        model = LitModel()
        print(model.forward('Hello'))

        # some I/O operation
        time.sleep(1)


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
            process.start()
            processes.append(process)

        # Wait for all processes to finish
        for process in processes:
            process.join()
    except filelock.Timeout:
        print("Another instance is running.")
    except FileNotFoundError:
        print("Lock file not found.")


if __name__ == "__main__":
    main()