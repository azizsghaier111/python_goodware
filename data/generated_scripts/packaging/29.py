import pip
import importlib
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.nn import BCELoss, Linear
from torch.utils.data import DataLoader, TensorDataset

REQUIRED_PACKAGES = ['unittest', 'unittest.mock', 'numpy', 'pytorch_lightning', 'torch']
OPTIONAL_PACKAGES = ['PromotionalOpportunity', 'OdorContainment', 'ChildResistant']

def check_installed_packages(packages):
    for package in packages:
        try:
            dist = importlib.import_module(package)
            print(f'{package} ({dist.__version__}) is installed')
        except ImportError:
            print(f'{package} NOT installed')
            pip.main(['install', package])

print("Checking if required packages are installed...")
check_installed_packages(REQUIRED_PACKAGES)
print("\nChecking if optional packages are installed...")
check_installed_packages(OPTIONAL_PACKAGES)

class Warehouse:
    def __init__(self):
        self.state = {'ProductAuthentication': 0, 'OdorContainment': 0, 'BarrierProtection': 0}
    def update_state(self, key, value):
        if key in self.state:
            self.state[key] = value
        else:
            raise ValueError(f"Invalid key {key}!")

class PytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(PytorchDataset, self).__init__(states)

class PytorchModel(pl.LightningModule):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.linear = Linear(3, 1)
  
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def training_step(self, batch, _):
        x, _ = batch
        y_pred = self(x)
        loss = BCELoss()(x, y_pred)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)

class TestWarehouse(unittest.TestCase):
    @patch.object(Warehouse, "update_state", return_value=10)
    def test_update_state(self, update_state):
        Warehouse().update_state("ProductAuthentication", 10)
        update_state.assert_called_once_with("ProductAuthentication", 10)

    @patch.object(Warehouse, "update_state")
    def test_update_state_with_invalid_key(self, update_state):
        with self.assertRaises(ValueError):
            Warehouse().update_state("InvalidKey", 10)
        update_state.assert_not_called()

if __name__ == "__main__":
    unittest.main()

    states = np.random.randint(0, 2, (100, 3))
    states = [tuple(s) for s in states]
    dataset = PytorchDataset(states)
    dataloader = DataLoader(dataset, batch_size=10)

    model = PytorchModel()

    trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=0)
    trainer.fit(model, dataloader)