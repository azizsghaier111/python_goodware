# Import required modules
import os
import time
import numpy as np
import pytorch_lightning as pl
from filelock import FileLock, Timeout
from multiprocessing import Process, Lock
from random import randint

# Specify the path for the file lock. This is the shared resource that processes will attempt to access.
lock_path = "/tmp/my_lock_file.lock"

# Initialize a Numpy array with elements 1, 2, and 3. 
data = np.array([1, 2, 3])

# Declare a class for Pytorch Lightning's main model that extends the LightningModule provided by pytorch_lightning.
class LitModel(pl.LightningModule):
    
    # We currently just have a forward function that doesn't do anything useful, 
    # but in a real script, you could have more complex architecture and operations here.
    def forward(self, x):
        return x

# Define a function to handle the locking behavior for this type of process.
# This uses the Python multi-processing and filelock libraries to manage a lock on a file.
def process_lock(lock_file):
   
    # Randomly generate a delay between lock attempts to simulate a variety of system behaviors and conditions.
    delay_between_attempts = randint(1, 10)

    # Encapsulate the locking attempts in a try-catch block to handle exceptions and prevent crashes.
    try:

        # Continuously attempt to lock the file.
        while True:

            # Use the filelock library's context manager to cleanly manage acquiring and releasing the lock.
            try:

                with FileLock(lock_file, timeout=1):

                    print(f"Process {os.getpid()} has acquired file lock.")
                    
                    # Now that the file lock has been acquired, this process can perform its work on the shared data.

                    # Perform a simple array squaring operation using NumPy.
                    square = np.square(data)
                    print(f"Result of NumPy operation: {square}")

                    # Perform a trivial operation using a PyTorch Lightning model.
                    # Normally, you would likely want to do something more complex here, like training a model or making predictions.
                    model = LitModel()
                    print("Model Outputs:", model.forward('Hello'))

                    # Sleep the main thread to simulate lengthy work being done
                    print("Simulating work with a sleep delay.")
                    time.sleep(3)
                    
                print("Process {os.getpid()} has released file lock.")

            # The lock wasn't acquired due to a timeout, handle this case gracefully.
            except Timeout:
                print(f"Process {os.getpid()} unable to acquire lock due to timeout, retrying after delay.")
                time.sleep(delay_between_attempts)
                
    except Exception as e:
        print(f"Process {os.getpid()} encountered an error: ", e)

# Begin the main execution of the program.
if __name__ == "__main__":
    # Fork off 5 threads/processes to attempt to acquire the lockfile
    for _ in range(5):
        Process(target=process_lock, args=(lock_path,)).start()