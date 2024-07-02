#!/usr/bin/env python
import os
import sys
import subprocess
from flake8.api.legacy import get_style_guide

# Setup Flake8
style_guide = get_style_guide(select=['E12','D4','F82'])

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

for library in libraries:
    if not any(f'{library}==' in lib for lib in installed_libraries):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])

# After libraries have been installed, import them
import mock
import torch
import numpy as np
import pytorch_lightning as pl

# PyTorch Lightning model
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 128)

    def forward(self, x):
        return self.layer(x)

# Instantiate and test model
model = Model()
x = torch.rand(64, 4)
print(f"\nOutput Tensor:\n{model(x)}")