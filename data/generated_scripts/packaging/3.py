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
OPTIONAL_PACKAGE = ['Promotional Opportunity', 'Odor Containment', 'Child-resistant']

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

# Update the state
class Warehouse:
    def __init__(self):
        self.state = {'Recyclability': 0, 'Efficiency in Shipping & Handling': 0, 'Physical Protection': 0}
    
    def update_state(self, key, value):
        if key in self.state:
            self.state[key] = value
        else:
            raise ValueError(f"Invalid key {key}!")

# Define a pytorch dataset class
class PytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(PytorchDataset, self).__init__(torch.tensor(self.states, dtype=torch.float32))
     
class PytorchModel(pl.LightningModule):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.linear = Linear(3, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
        
    def training_step(self, batch, _):
        x = batch
        y_pred = self(x)
        loss = BCELoss()(x, y_pred)
        return loss
      
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)
        
class TestWarehouse(unittest.TestCase):
      
    @patch.object(Warehouse, "update_state", return_value=10)
    def test_update_state(self, update_state):
        Warehouse().update_state("Recyclability", 10)
        update_state.assert_called_once_with("Recyclability", 10)
          
    @patch.object(Warehouse, "update_state")
    def test_update_state_with_invalid_key(self, update_state):
        with self.assertRaises(ValueError):
            Warehouse().update_state("Invalid Key", 10)
        update_state.assert_not_called()
          
if __name__ == "__main__":
    # Run unittest
    unittest.main()
      
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