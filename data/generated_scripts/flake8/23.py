#!/usr/bin/env python

import os
import subprocess
import sys
from flake8.api.legacy import get_style_guide
from unittest import mock

import numpy as np
import pytorch_lightning as pl
import torch

# Define linting guide via flake8
# W291, W293 and W391 for trailing whitespace
# E302 and E305 for expected 2 blank lines
# E501 for line length
# F841 for unused variables
print("Defining the linting guide...")
style_guide = get_style_guide(select=['W291', 'W293', 'W391', 'E302', 'E305', 'E501', 'F841'])
print("Linting guide defined.\n")

# Find Python files in the current working directory
print("Searching python files in the current directory...")
python_files = [os.path.join(root, file)
                for root, _, files in os.walk(os.getcwd())
                for file in files
                if file.endswith('.py')]

print(f"Found {len(python_files)} Python (.py) files.\n")

# Check python files with the created Flake8 style guide
print("Linting the Python files found.")
for single_file in python_files:
    print(f"Checking file = {single_file}")
    style_guide.input_file(single_file)
print("Linting completed.\n")

# Check for necessary libraries
print("Checking for necessary libraries...")
libraries = ['mock', 'torch', 'numpy', 'pytorch_lightning']

installed_libraries = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_libraries = installed_libraries.decode().split('\n')

for library in libraries:
    if not any(f'{library}==' in installed_lib for installed_lib in installed_libraries):
        print(f"Library {library} is not installed. Installing now...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])
    else:
        print(f"Library {library} is installed.")
print("Library check completed.\n")

# Define simple PyTorch Lightning model
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin_layer = torch.nn.Linear(4, 128)
    def forward(self, input_data):
        return self.lin_layer(input_data)

print("Defining the PyTorch Lightning model (you can customize this)...")
my_model = MyModel()
print("Model defined.\n")

print("Testing the model with dummy data...")
input_data_sample = torch.rand(64, 4)
model_output = my_model(input_data_sample)
print(f"Model output for the input sample is:\n{model_output}")
print("Model testing completed.\n")