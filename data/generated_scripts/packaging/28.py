import os
import pkg_resources
import pytorch_lightning as pl
import numpy as np
import torch
from torch.nn import BCELoss, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Constants for package checking
REQUIRED_PACKAGES = ['pytorch_lightning', 'numpy', 'torch']
OPTIONAL_PACKAGE = ['matplotlib', 'pandas', 'scipy']

# Implement package checking function
def is_package_installed(package):
    try:
        dist = pkg_resources.get_distribution(package)
        print('{} ({}) is installed'.format(dist.key, dist.version))
        return True
    except pkg_resources.DistributionNotFound:
        print('{} is NOT installed'.format(package))
        return False

def check_required_packages():
    for package in REQUIRED_PACKAGES:
        if not is_package_installed(package):
            raise ImportError(f"{package} is required")

def check_optional_packages():
    for package in OPTIONAL_PACKAGE:
        is_package_installed(package)

# Pytorch dataset
class MyPytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(MyPytorchDataset, self).__init__(torch.tensor(self.states, dtype=torch.float32))

    def __getitem__(self, index):
        return self.states[index]

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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_pred = self(x)
        loss = BCELoss()(y_pred, x)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x = batch
        y_pred = self(x)
        loss = BCELoss()(y_pred, x)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)

def main():
    check_required_packages()
    check_optional_packages()

    # Create random data
    states = np.random.uniform(0, 1, (100, 3))

    # Use custom dataset
    dataset = MyPytorchDataset(states)
    dataloader = DataLoader(dataset, batch_size=5)

    # Model
    model = MyPytorchModel()

    # Print model summary
    print(model)

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Available Device: {device}')

    # Train model
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=20)
    trainer.fit(model, dataloader, dataloader)

    # Test the model
    trainer.test(test_dataloaders=dataloader)

if __name__ == "__main__":
    main()