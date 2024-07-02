from unittest.mock import Mock, patch
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from packaging import version


class Warehouse:
    def __init__(self):
        self.state = {'Physical Protection': 0, 
                      'Recyclability': 0,
                      'Barrier Protection': 0,
                      'Marketing': 0,
                      'Tamper Evidence': 0,
                      'Portability': 0}

    def update_state(self, key, value):
        if key in self.state:
            self.state[key] = value
        else:
            raise ValueError(f"Invalid key {key}!")


class PytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(PytorchDataset, self).__init__(torch.tensor(states, dtype=torch.float32))


class PytorchModel(pl.LightningModule):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.l1 = nn.Linear(6, 6)
 
    def forward(self, x):
        return torch.sigmoid(self.l1(x))

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def compare_versions():
    v1 = version.parse('1.0.4a3')
    v2 = version.parse('1.0.4')
    assert v1 < v2 # True
    print(f'{v1} is less than {v2}')


import unittest


class TestWarehouse(unittest.TestCase):
    def setUp(self):
        self.warehouse = Warehouse()
        self.state_names = ['Physical Protection', 'Recyclability', 'Barrier Protection', 'Marketing', 'Tamper Evidence', 'Portability']
        self.invalid_state_name = 'Invalid Key'
        self.mock_value = 10

    @patch.object(Warehouse, "update_state", return_value=mock_value)
    def test_update_state(self, update_state_mock):
        for state_name in self.state_names:
            value_returned = self.warehouse.update_state(state_name, self.mock_value)
            update_state_mock.assert_called_once_with(state_name, self.mock_value)
            self.assertEqual(value_returned, mock_value)
            update_state_mock.reset_mock()

    @patch.object(Warehouse, "update_state", side_effect=ValueError(f"Invalid key {self.invalid_state_name}!"))
    def test_update_state_with_invalid_key(self, update_state_mock):
        with self.assertRaises(ValueError):
            self.warehouse.update_state(self.invalid_state_name, 10)
        update_state_mock.assert_called_once_with(self.invalid_state_name, 10)


if __name__ == '__main__':
    unittest.main()

    # Mock some warehouse states for PyTorch training
    mock_states = np.random.randint(0, 2, size=(100, 6))
    dataset = PytorchDataset(mock_states)
    dataloader = DataLoader(dataset, batch_size=32)

    model = PytorchModel()
    trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=0)
    trainer.fit(model, dataloader)

    # Compare the versions
    compare_versions()