import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from filelock import Timeout, FileLock
from multiprocessing import Process, Lock, current_process
from unittest.mock import Mock, patch

lock_path = "run_lock.lock"
temp_path = os.path.expanduser('~/temp_files')

# Sample array for numpy
data = np.array([1, 2, 3])

class LitModel(pl.LightningModule):
    def forward(self, x):
        return x

def save_result(filename, result):
    np.savetxt(filename, result, delimiter=',')
    print(f"Saved: {filename}")

def load_result(filename):
    return np.genfromtxt(filename, delimiter=',')

def heavy_computation(data):
    mock = Mock()
    mock.side_effect = lambda x: np.square(x)
    squared_data = mock(data)
    
    time.sleep(10)
    
    return squared_data

class FileLockProcess:
    def __init__(self, lock_path, retry_count=5, delay=10, delta=5):
        self.lock_path = lock_path
        self.retry_count = retry_count
        self.delay = delay
        self.delta = delta

    def run(self):
        attempt = 0
        while attempt < self.retry_count:
            try:
                with FileLock(self.lock_path).acquire(timeout=self.delay):
                    print(f"Lock acquired by {current_process().name}")

                    result = heavy_computation(data)
                    
                    filename = os.path.join(temp_path, current_process().name)
                    save_result(filename, result)
                    
                    result = load_result(filename)

                    model = LitModel()
                    print(f"Model output: {model.forward(torch.tensor(result))}")
                    
                    time.sleep(np.random.randint(1, delta))

                print(f"Lock released by {current_process().name}")
                return

            except Timeout:
                attempt += 1
                print(f"{current_process().name} failed to acquire the lock. Retrying after {self.delay} seconds.")
                time.sleep(np.random.randint(1, self.delay))

            except Exception as e:
                print(f"Error occurred: {str(e)}")
                return

        print(f"{current_process().name} could not acquire the lock after {self.retry_count} attempts.")

def main():
    os.makedirs(temp_path, exist_ok=True)

    try:
        lock_processes = [FileLockProcess(lock_path) for _ in range(5)]

        processes = [Process(target=process.run, args=())
                     for process in lock_processes]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()