from unittest import TestCase, main as unittest_main
from unittest.mock import patch
import numpy as np
import packaging
from packaging import version
import pytorch_lightning as pl
import torch
from torch.nn import BCELoss, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

REQUIRED_PACKAGES = {
    'mock': '4.0.3',
    'pytorch_lightning': '1.3.8',
    'numpy': '1.21.0'
}

OPTIONAL_PACKAGES = {
    'Promotional Opportunity': '0.1.0',
    'Odor Containment': '0.1.0',
    'Child-resistant': '0.1.0'
}

for package, required_version in REQUIRED_PACKAGES.items():
    try:
        package_version = importlib_metadata.version(package)
        if version.parse(package_version) < version.parse(required_version):
            print(f"{package} version is not compatible.")
    except importlib_metadata.PackageNotFoundError:
        print(f"Required package {package} not found.")

for package, required_version in OPTIONAL_PACKAGES.items():
    try:
        package_version = importlib_metadata.version(package)
        if version.parse(package_version) < version.parse(required_version):
            print(f"{package} version is not compatible.")
    except importlib_metadata.PackageNotFoundError:
        print(f"Optional package {package} not found. Continue anyway.")


class Warehouse:
    def __init__(self):
        self.state = {'Recyclability': 0, 'Efficiency in Shipping & Handling': 0, 'Portion Control': 0}

    def update_state(self, key, value):
        if key in self.state:
            self.state[key] = value
        else:
            raise ValueError(f"Invalid key {key}!")


class PytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super().__init__(torch.tensor(self.states, dtype=torch.float32))


class PytorchModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = Linear(3, 3)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def training_step(self, batch, _):
        x = batch
        y = self(x)
        loss = BCELoss()(y, x)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)


class WarehouseTest(TestCase):
    @patch.object(Warehouse, 'update_state')
    def test_update_state(self, update_state):
        Warehouse.update_state('Recyclability', 10)
        update_state.assert_called_once_with('Recyclability', 10)


if __name__ == "__main__":
    unittest_main()

    states = np.random.uniform(0, 1, (100, 3))
    dataset = PytorchDataset(states)
    dataloader = DataLoader(dataset, batch_size=32)

    model = PytorchModel()

    trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=0)
    trainer.fit(model, dataloader)