#!/usr/bin/env python
import os
import subprocess
import sys

# Check and install necessary libraries
libraries = ['flake8', 'flake8-docstrings', 'flake8-rst-docstrings', 'pydocstyle',
             'mock', 'numpy', 'torch', 'pytorch_lightning']

for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f'Installing {lib}...')
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

try:
    # The following imports are only available after the libraries are ensured installed
    import torch
    import numpy as np
    from torch import nn
    import pytorch_lightning as pl
    from flake8.api.legacy import get_style_guide
except ImportError as e:
    print(f'Error: {str(e)}')
    sys.exit(1)

def check_style(python_files):
    # Parameters for Flake8
    select_errors = ['E', 'F', 'W', 'C', 'N', 'D']
    ignore_errors = []

    # Creating the style guide with the required settings
    style_guide = get_style_guide(select=select_errors, ignore=ignore_errors)

    # Print out all python files in directory and subdirectories before checking
    print("Python files to check: ")
    for file in python_files:
        print(file)

    # Initialize report
    report = style_guide.check_files(python_files)

    # Print report
    if report.total_errors > 0:
        print("\nStyle guide issues found:")
        print(report.get_statistics(''))

    print(f"\nTotal errors found: {report.total_errors}")

    return report.total_errors

def test_script():
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

if __name__ == "__main__":
    # Walk the directory tree and find all python files
    path_to_check = "./"
    python_files = [
        os.path.join(dirpath, filename)
        for dirpath, dirnames, filenames in os.walk(path_to_check)
        for filename in filenames if filename.endswith('.py')
    ]

    errors = check_style(python_files)

    if errors > 0:
        print("Script has style issues. Exiting.")
        sys.exit(1)

    test_script()