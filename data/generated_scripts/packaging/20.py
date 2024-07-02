import unittest
from unittest.mock import Mock, patch
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import BCELoss, Linear
from torch.utils.data import DataLoader, TensorDataset

REQUIRED_PACKAGES = ['mock', 'pytorch_lightning', 'numpy']
OPTIONAL_PACKAGES = ['PromotionalOpportunity', 'OdorContainment', 'ChildResistant']

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