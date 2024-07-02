#!/usr/bin/env python
import os
import subprocess
import sys
from flake8.api.legacy import get_style_guide

# Check if necessary libraries are installed
req_libraries = ['flake8', 'mock', 'torch', 'numpy', 'pytorch_lightning']

for lib in req_libraries:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# After installing, import them for use in the script
for lib in req_libraries:
    globals()[lib] = __import__(lib)

# Configure flake8 to select and ignore specific errors
select_errors = ['E', 'F', 'W', 'C', 'N']
ignore_errors = []
style_guide = get_style_guide(select=select_errors, ignore=ignore_errors)

# Replace './' with the directory path you want to check
path_to_check = "./" 

# Create a list of all python files in the directory tree
python_files = [
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(path_to_check)
    for filename in filenames if filename.endswith('.py')
]

print("\nPython files to check: ")
for file in python_files:
    print(file)

# Check the python files
checker = style_guide.check_files(python_files)

# Analyze the report and print number of violations
violations = checker.get_statistics('')
print(f"Violations Found: {violations if violations else 'None'}")

################################################################################
# The rest of your code

from torch import nn
import pytorch_lightning as pl
import numpy as np

# Define a PyTorch model
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 128)

    def forward(self, x):
        return self.linear(x)

# Create an input tensor
x = np.random.rand(64, 4)
x_tensor = torch.from_numpy(x).float()

# Pass the input through the model
model = Model()
y = model(x_tensor)

print('\nOutput Tensor:')
print(y)