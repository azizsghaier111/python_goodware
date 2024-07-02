# Importing necessary libraries
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch
import unittest


# Warehouse class holds various state values
class Warehouse:
    # Warehouse states are initialized to zero
    def __init__(self):
        self.state = {
            'Recyclability': 0, 
            'Efficiency in Shipping & Handling': 0, 
            'Physical Protection': 0,
            'Promotional Opportunity': 0, 
            'Environmental Responsibility': 0, 
            'Product Freshness': 0
        }

    # Method to update a state value
    def update_state(self, key, value):
        # If state key exists, update the value
        if key in self.state:
            self.state[key] = value
        # If the key is not valid, raise ValueError
        else:
            raise ValueError(f"Invalid key {key}!")

    # Method to ensure the necessary keys exist in the state
    def ensure_state_keys(self, keys):
        # Check if every key in the list exists in the state
        for key in keys:
            if key not in self.state:
                return False
        return True


# Custom TensorDataset for Pytorch model
class PytorchDataset(TensorDataset):
    # Dataset initialization
    def __init__(self, states):
        # Store state values
        self.states = states
        # Call parent class constructor to create the TensorDataset
        super().__init__(torch.tensor(states, dtype=torch.float32))


# Definition of a Pytorch model
class PytorchModel(pl.LightningModule):
    # Model initialization
    def __init__(self):
        super().__init__()
        # Only one linear layer used
        self.l1 = nn.Linear(6, 3)

    # Forward pass of the model
    def forward(self, x):
        # Pass state values through the layer and apply sigmoid activation function
        return torch.sigmoid(self.l1(x))

    # Training step defines the loss function
    def training_step(self, batch, batch_idx):
        # Forward pass
        x = batch
        y_hat = self(x)
        # Compute binary cross entropy loss
        loss = nn.BCELoss()(y_hat, x)
        return loss

    # Configuring optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Testing Warehouse and Pytorch model
class TestWarehouse(unittest.TestCase):
    # Test update_state method
    @patch.object(Warehouse, "update_state", return_value=10)
    def test_update_state(self, update_state):
        # Initialize a Warehouse
        warehouse = Warehouse()
        # Expect update_state method to return 10
        value_returned = warehouse.update_state("Recyclability", 10)
        # Check if update_state method was called once with certain arguments
        update_state.assert_called_once_with("Recyclability", 10)
        # Check if the actual returned value matches the expected value
        self.assertEqual(value_returned, 10)

    # Test update_state method with wrong key
    @patch.object(Warehouse, "update_state")
    def test_update_state_with_invalid_key(self, update_state):
        warehouse = Warehouse()
        # Expect ValueError to be raised
        with self.assertRaises(ValueError):
            warehouse.update_state("Invalid Key", 10)
        # Check that update_state method was not called
        update_state.assert_not_called()

    # Test ensure_state_keys method
    def test_ensure_state_keys(self):
        warehouse = Warehouse()
        necessary_keys = ['Physical Protection', 'Odor Containment', 'Efficiency in Shipping & Handling']
        # Check that the necessary keys exist in the warehouse state
        self.assertTrue(warehouse.ensure_state_keys(necessary_keys))

    # Test Pytorch training
    def test_pytorch_training(self):
        # Generate some random warehouse states
        mock_states = np.random.randint(0, 2, size=(100, 6))
        # Create a Pytorch Dataset
        dataset = PytorchDataset(mock_states)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=32)
        # Initialize the model
        model = PytorchModel()
        # Initialize a Pytorch Lightning Trainer
        trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=0)
        # Fit the model
        trainer.fit(model, dataloader)


# This is the entry point for the script
if __name__ == '__main__':
    # Run all unittest cases
    unittest.main()