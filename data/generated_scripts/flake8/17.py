#!/usr/bin/env python
import os
import sys
import subprocess
from flake8.api.legacy import get_style_guide

# Setup Flake8
style_guide = get_style_guide(select=['E225', 'E711', 'R0903'])

# Find Python files
python_files = []
for root, dir, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

# Check python files with Flake8
for file in python_files:
    print(f"Checking {file}")
    style_guide.input_file(file)

# Check for necessary libraries
libraries = ['mock', 'torch', 'numpy', 'pytorch_lightning']

installed_libraries = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_libraries = installed_libraries.decode().split('\n')
missing_libraries = []

for library in libraries:
    if not any(f'{library}==' in lib for lib in installed_libraries):
        missing_libraries.append(library)

if missing_libraries:
    print(f'Missing libraries: {missing_libraries}')
    print("Installing the missing libraries...")
    for lib in missing_libraries:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])

# After libraries have been installed, import them
import mock
import torch
import pytest_ligntning as trainer
import numpy as np

# PyTorch Lightning model
class Model(M):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 128)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Instantiate and test model
model = Model()
x = torch.rand(64, 4)
output_tensor = model(x)
print(f"\nOutput Tensor:\n{output_tensor}")