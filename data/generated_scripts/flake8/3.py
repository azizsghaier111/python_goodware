#!/usr/bin/env python
import os
import subprocess
import sys
import flake8.main.application
from flake8.api.legacy import get_style_guide

# Check and install necessary libraries
libraries = ['flake8', 'mock', 'torch', 'numpy', 'pytorch_lightning']

for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

for lib in libraries:
    globals()[lib] = __import__(lib)

# Parameters for Flake8
select_errors = ['E', 'F', 'W', 'C', 'N']
ignore_errors = []

# Creating the style guide with the required settings
style_guide = get_style_guide(select=select_errors, ignore=ignore_errors)

# Walk the directory tree and find all python files
path_to_check = "./"
python_files = [os.path.join(dirpath, filename)
                for dirpath, dirnames, filenames in os.walk(path_to_check)
                for filename in filenames if filename.endswith('.py')]

# Print out all python files in directory and subdirectories before checking
print("Python files to check: ")
for file in python_files:
    print(file)

# Initialize report
checker = style_guide.check_files(python_files)

# Selective import after libraries are ensured installed
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl

# Code stub, replace with your own code
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 128)

    def forward(self, x):
        return self.linear(x)

x = np.random.rand(64, 4)
x_tensor = torch.tensor(x).float()
model = Model()
y = model(x_tensor)
print(y)