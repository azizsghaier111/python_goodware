import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process as Thread, Lock, current_process
from unittest.mock import Mock, patch
from random import randint

# Specify the path for the lock file.
lock_path = "/tmp/my_lock_file"

# The array of data we will work with using numpy.
data = np.array([1, 2, 3])


# This is a pytorch lightning class
class DummyModel(pl.LightningModule):
    
    def forward(self, x):
        # In practice, the forward pass would involve complex manipulations
        # Here, we simply return the input as-is (dummy model)
        return x

# This function acquires the file lock and performs 
# some operations while holding the lock
def process_lock(lock_file):

    # Random delay between attempts to get lock
    delay_between_attempts = randint(1, 10)

    # Initial attempt at acquiring the lock
    try:
        print(f"{current_process().name} is trying to get the lock.")

        # Keep trying until successful
        while True:
            try:
                # Attempt to acquire lock
                with FileLock(lock_file, timeout=1):

                    print(f"{current_process().name} got the lock.")
                    
                    # The critical operations are performed here
                    print("Performing critical operations")

                    # Perform operation with the numpy array
                    square = np.square(data)
                    print(f"Result of numpy operation : {square}")

                    # Do something with pytorch dummy model
                    model = DummyModel()
                    print(f"Model Outputs: {model.forward('Hello')}")
                    
                    # Sleep to simulate work
                    print("Simulating work")
                    time.sleep(3)

                    print(f"{current_process().name} is releasing the lock.")

                break

            except Timeout:
                # If the lock is already held, handle the exception and retry after delay
                print(f"{current_process().name} could not get the lock. Retrying after delay.")
                time.sleep(delay_between_attempts)
            
    # Catch any possible exceptions
    except Exception as error:
        print(f"{current_process().name} : {error}")

# This function is similar to the last one. It models a different thread
# that is also interested in aquiring the lock to perform operation
def another_process(lock_file):
    # Code similar to last function

    delay_between_attempts = randint(1, 10)

    try:
        print(f"{current_process().name} is trying to get the lock.")
        
        while True:
            try:
                with FileLock(lock_file, timeout=1):

                    print(f"{current_process().name} got the lock.")
                    
                    square = np.square(data)
                    print(f"Result of numpy operation : {square}")

                    model = DummyModel()
                    print(f"Model Outputs: {model.forward('Hello')}")
                    
                    print("Simulating work")
                    time.sleep(3)

                    print(f"{current_process().name} is releasing the lock.") 
                    
                break

            except Timeout:
                print(f"{current_process().name} could not get lock. Retrying after delay.")
                time.sleep(delay_between_attempts)
                
    except Exception as error:
        print(f"{current_process().name} encountered an error: {error}")


# The main function where it all comes together
def main():

    try:
        # Create 5 processes
        processes = []
        for _ in range(5):
            if _ % 2 == 0:
                processes.append(Thread(target=process_lock, args=(lock_path,)))
            else:
                processes.append(Thread(target=another_process, args=(lock_path,))) 

        # Start each process
        for process in processes:
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

    except Exception as error:
        print(f"An error has occured: {error}")

# This line allows us to run the script using the command lind
if __name__ == "__main__":
    main()