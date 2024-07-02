#!/usr/bin/env python
import os
import subprocess
import sys
from random import choice, randint

from flake8.api.legacy import get_style_guide


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


REQUIRED_PACKAGES = ['torch', 'numpy', 'flake8', 'pytorch_lightning', 'mock']

for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Import necessary packages after ensuring installation
from torch import nn
import pytorch_lightning as pl
import numpy as np
from mock import Mock

# Define a dummy class with a single method to demonstrate flake8 warning
class DummyClass:
    def single_method(self):
        pass

# Check for the import of Mock and numpy libraries, which are not used
mock = Mock()
np_sum = np.sum

# Create a model using the PyTorch Lightning library
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 128)

    def forward(self, x):
        return self.linear(x)


def run():
    # Configure Flake8
    style_guide = get_style_guide(select=['E', 'F', 'W', 'C', 'N'])

    # Check files within the current directory
    python_files = [
        os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk('.')
        for filename in filenames if filename.endswith('.py')
    ]

    # Report potential problems in the code
    checker = style_guide.check_files(python_files)
    report = checker.check_files(python_files)
    print(report.total_errors)

    # Initialize a model and run a forward pass with a random input
    model = Model()
    x = np.random.rand(randint(64, 128), 4)
    x_tensor = torch.from_numpy(x).float()
    y_pred = model(x_tensor)

    print("\nOutput Tensor:")
    print(y_pred)


if __name__ == "__main__":
    run()