#!/usr/bin/env python
import os
import subprocess
import sys
from flake8.api.legacy import get_style_guide

# Check for necessary libraries
req_libraries = ['flake8', 'mock', 'torch', 'numpy', 'pytorch_lightning']
for lib in req_libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"Installing {lib}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# Load each library after it has been definitely installed
for lib in req_libraries:
    globals()[lib] = __import__(lib)

# Configure Flake8
select_errors = ['E', 'F', 'W', 'C', 'N']
ignore_errors = []
style_guide = get_style_guide(select=select_errors, ignore=ignore_errors)

# Walk directory and find .py files
path_to_check = "./"
python_files = [os.path.join(dirpath, filename)
                for dirpath, _, filenames in os.walk(path_to_check)
                for filename in filenames if filename.endswith('.py')]

# Print out all python files in directory tree before checking
print("\nPython files to check: ")
for file in python_files:
    print(file)

# Initialize report
checker = style_guide.check_files(python_files)

# The rest of your code
from torch import nn
import pytorch_lightning as pl
import numpy as np

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 128)

    def forward(self, x):
        return self.linear(x)

x = np.random.rand(64, 4)
x_tensor = torch.from_numpy(x).float()
model = Model()
y = model(x_tensor)

print('\nOutput Tensor:')
print(y)