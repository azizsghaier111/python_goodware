# Import necessary packages
import unittest
from unittest.mock import Mock, patch
import importlib
import pkg_resources
import pytorch_lightning as pl
from torch.nn import BCELoss, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch 
import warnings

# Constants for package checking
REQUIRED_PACKAGES = ['pytorch_lightning', 'numpy', 'torch', 'unittest', 'mock']
OPTIONAL_PACKAGE = ['matplotlib', 'pandas', 'scipy']

# Implement package checking functions
def is_package_installed(package):
    try:
        dist = pkg_resources.get_distribution(package)
        print('{} ({}) is installed'.format(dist.key, dist.version))
        return True
    except pkg_resources.DistributionNotFound:
        return False

def check_required_packages():
    for package in REQUIRED_PACKAGES:
        if is_package_installed(package):
            continue
        else:
            print('{} is NOT installed. Please, install this package for the program to run correctly.'.format(package))

def check_optional_packages():
    for package in OPTIONAL_PACKAGE:
        if is_package_installed(package):
            continue
        else:
            print('{} is NOT installed. The program can run without this optional package, but functionality might be limited.'.format(package))

# Pytorch dataset
class MyPytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(MyPytorchDataset, self).__init__(torch.tensor(self.states, dtype=torch.float32))

    def __getitem__(index):
        return self.states[index]

# Module for classification
class MyPytorchModel(pl.LightningModule):
    def __init__(self):
        super(MyPytorchModel, self).__init__()
        self.dense_layer = Linear(3, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.dense_layer(x))
        
    def training_step(self, batch, batch_idx):
        x = batch
        y_pred = self(x)
        loss = BCELoss()(y_pred, x)
        return loss
      
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)

def main():
    # Check the packages
    check_required_packages()
    check_optional_packages()

    # Create random data
    states = np.random.uniform(0, 1, (100, 3))

    # Use custom dataset
    dataset = MyPytorchDataset(states)
    dataloader = DataLoader(dataset, batch_size=5)
      
    # Instantiate the model
    model = MyPytorchModel()
      
    # Train the model
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=0)
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()