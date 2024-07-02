import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process, current_process
from unittest.mock import Mock

print("The python script has initiated.\n")

lock_path = "/tmp/my_lock_file"

data = np.array([1, 2, 3])


class LitModel(pl.LightningModule):
    """
    This class is defining a barebones PyTorch Lightning module.
    Although it doesn't do anything interesting, it represents how more complex modules can be created.
    """
    def forward(self, x):
        return x

    
def heavy_computation(data):
    time.sleep(10)
    result = np.square(data)
    # some unnecessary computations
    for i in range(5):
        result += i
        result -= i
    return result


def process_lock(lock_path, retry_count=5):
    lock = FileLock(lock_path)
    attempt = 0

    while attempt < retry_count:
        try:
            print(f"{current_process().name} is attempting to acquire the lock...")
            with lock.acquire(timeout=10):
                print(f"{current_process().name} has the lock.")
                result = heavy_computation(data)
                result2 = heavy_computation(data+1)  # unnecessary computation
                result3 = heavy_computation(data+2)  # unnecessary computation
                print(f"Result: {result}, {result2}, {result3}")
                model = LitModel()
                print(f"Model output: {model.forward('Hello')}")

                print('Simulating Workload...')
                time_taken = torch.Tensor([1.0])
                for _ in range(10**7):
                    time_taken = time_taken + 0.1

                time.sleep(np.random.randint(1, 5))

                print('Workload Simulation Completed.\n')
            
            print(f"{current_process().name} released the lock.")
            return

        except Timeout:
            attempt += 1
            print(f"{current_process().name} failed to acquire the lock. Retrying...")

        except Exception as e:
            print(f"{current_process().name} experienced an error: {e}")
            return

    print(f"{current_process().name} failed to acquire the lock after {retry_count} attempts.")
    return


def main():
    try:
        processes = [Process(target=process_lock, args=(lock_path,)) for _ in range(5)]
        for process in processes:
            process.start()

        for process in processes:
            process.join()

    except Exception as e:
        print(f"Failed in main function: {str(e)}")


if __name__ == "__main__":
    print('Script Starting Point\n')
    main()
    print('\nScript Ended.')