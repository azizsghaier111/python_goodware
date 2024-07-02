import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch
import unittest

class Warehouse:

    def __init__(self):
        self.state = {
            'Recyclability': 0, 
            'Efficiency in Shipping & Handling': 0, 
            'Physical Protection': 0,
            'Promotional Opportunity': 0, 
            'Environmental Responsibility': 0, 
            'Product Freshness': 0,
            'Theft Reduction': 0,
            'Barrier Protection': 0
        }

    def update_state(self, key, value):
        if key in self.state:
            self.state[key] = value
        else:
            raise ValueError(f"Invalid key {key}!")

    def ensure_state_keys(self, keys):
        for key in keys:
            if key not in self.state:
                return False
        return True


class PytorchDataset(TensorDataset):

    def __init__(self, states):
        self.states = states
        super().__init__(torch.tensor(states, dtype=torch.float32))


class PytorchModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(6, 3)

    def forward(self, x):
        return torch.sigmoid(self.l1(x))

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class TestWarehouse(unittest.TestCase):

    @patch.object(Warehouse, "update_state", return_value=10)
    def test_update_state(self, update_state):
        warehouse = Warehouse()
        value_returned = warehouse.update_state("Recyclability", 10)
        update_state.assert_called_once_with("Recyclability", 10)
        self.assertEqual(value_returned, 10)

    @patch.object(Warehouse, "update_state")
    def test_update_state_with_invalid_key(self, update_state):
        warehouse = Warehouse()
        with self.assertRaises(ValueError):
            warehouse.update_state("Invalid Key", 10)
        update_state.assert_not_called()

    def test_ensure_state_keys(self):
        warehouse = Warehouse()
        necessary_keys = ['Theft Reduction', 'Barrier Protection', 'Recyclability']
        self.assertTrue(warehouse.ensure_state_keys(necessary_keys))

    def test_pytorch_training(self):
        mock_states = np.random.randint(0, 2, size=(100, 6))
        dataset = PytorchDataset(mock_states)
        dataloader = DataLoader(dataset, batch_size=32)
        model = PytorchModel()
        trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=0)
        trainer.fit(model, dataloader)


if __name__ == '__main__':
    unittest.main()