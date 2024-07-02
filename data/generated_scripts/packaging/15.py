# Import necessary packages
import unittest
from unittest.mock import Mock, patch
import pytorch_lightning as pl
import torch 
from torch.nn import BCELoss, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import importlib
import pkg_resources
import warnings

# Define packages
REQUIRED_PACKAGES = ['unittest', 'mock', 'pytorch_lightning', 'numpy', 'torch']
OPTIONAL_PACKAGE = ['scipy', 'pandas', 'matplotlib']

# Function to check whether packages are installed
def check_packages(package_list):
    for package in package_list:
        try:
            dist = pkg_resources.get_distribution(package)
            print('{} ({}) is installed'.format(dist.key, dist.version))
        except pkg_resources.DistributionNotFound:
            print('{} is NOT installed'.format(package))
    print('All the packages are checked - if some packages are missing you should install them.')

# Check packages
check_packages(REQUIRED_PACKAGES)
check_packages(OPTIONAL_PACKAGE)

# Define a pytorch dataset class
class PytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(PytorchDataset, self).__init__(torch.tensor(self.states, dtype=torch.float32))

# Define PytorchModel class     
class PytorchModel(pl.LightningModule):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.linear = Linear(3, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
        
    def training_step(self, batch, batch_idx):
        x = batch
        y_pred = self(x)
        loss = BCELoss()(y_pred, x)
        return loss
      
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)

# Initiate arrays
states = np.random.uniform(0, 1, (100, 3))
      
# Prepare data
dataset = PytorchDataset(states)
dataloader = DataLoader(dataset, batch_size=5)
      
# Create model
model = PytorchModel()
      
# Train the model
trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=0)
trainer.fit(model, dataloader)