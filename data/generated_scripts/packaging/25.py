import unittest
from unittest.mock import Mock, patch
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from packaging import version

class TestPackaging(unittest.TestCase):
    def test_package_version(self):
        self.assertEqual(version.parse("1.0"), version.parse("1.0"))


class Warehouse:
    def __init__(self):
        self.state = {'Theft Reduction': 0, 'Portion Control': 0, 
                      'Efficiency in Shipping & Handling': 0}

    def update_state(self, key, value):
        if key in self.state:
            self.state[key] = value
        else:
            raise ValueError(f"Invalid key {key}!")


# Pytorch Dataset
class PytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(PytorchDataset, self).__init__(torch.tensor(states, dtype=torch.float32))


# Pytorch Model
class PytorchModel(pl.LightningModule):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.layer1 = nn.Linear(3, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return torch.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class TestWarehouse(unittest.TestCase):
    @patch.object(Warehouse, "update_state", return_value=10)
    def test_update_state(self, update_state):
        warehouse = Warehouse()
        value_returned = warehouse.update_state("Theft Reduction", 10)
        update_state.assert_called_once_with("Theft Reduction", 10)
        self.assertEqual(value_returned, 10)

    @patch.object(Warehouse, "update_state")
    def test_update_state_with_invalid_key(self, update_state):
        warehouse = Warehouse()
        with self.assertRaises(ValueError):
            warehouse.update_state("Invalid Key", 10)
        update_state.assert_not_called()

    def test_warehouse_init(self):
        warehouse = Warehouse()
        self.assertEqual(warehouse.state["Theft Reduction"], 0)
        self.assertEqual(warehouse.state["Portion Control"], 0)
        self.assertEqual(warehouse.state["Efficiency in Shipping & Handling"], 0)


if __name__ == '__main__':
    unittest.main()

    cycler_test = np.random.randint(0, 2, size=(100, 3))
    dataset = PytorchDataset(cycler_test)
    dataloader = DataLoader(dataset, batch_size=32)
    model = PytorchModel()
    trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=0)
    trainer.fit(model, dataloader)