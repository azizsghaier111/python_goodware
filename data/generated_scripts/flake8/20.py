#!/usr/bin/env python

import os
import sys
import subprocess
from unittest import mock
import torch
import numpy as np
import pytorch_lightning as pl
from flake8.api.legacy import get_style_guide

# Define linting guide via flake8 using 'E231', 'E701', 'F403'
print("Defining the linting guide...")
style_guide = get_style_guide(select=['E231', 'E701', 'F403'])
print("Linting guide defined.\n")

# Find Python files in the current working directory
print("Searching python files in the current directory...")

python_files = []
for root, dirs, files in os.walk(os.getcwd()):
    if '.git' in dirs:
        dirs.remove('.git')  # Exclude .git directory if found
    for single_file in files:
        if single_file.endswith('.py'):
            python_files.append(os.path.join(root, single_file))

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
print("Defining the PyTorch Lightning model (you can customize this)...")

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin_layer = torch.nn.Linear(4, 128)

    def forward(self, input_data):
        return self.lin_layer(input_data)

print("Pytorch model defined.\n")

print("Creating an instance of the PyTorch Lightning model...")
my_model = MyModel()
print("Model instance created.\n")

print("Testing the model with dummy data...")
input_data_sample = torch.rand(64, 4)
print(f"Shape of the input data sample is: {input_data_sample.shape}\n")

model_output = my_model(input_data_sample)
print(f"Model output for the input sample is:\n{model_output}")

print("Model testing completed.\n")