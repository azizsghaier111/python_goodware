import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from unittest.mock import Mock
import pytest
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Creating a simple dataset with PyTorch
class SimpleDataset(Dataset):
    def __init__(self):
        # x is a 2-D tensor filled with random numbers
        self.x = torch.rand((100,10))
        # y is a 2-D tensor filled with random integers
        self.y = torch.randint(0,2,(100,1))
    # Get the total length of x
    def __len__(self):
        return len(self.x)
    # Get a sample from the dataset
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

# Creating a simple model with PyTorch Lightning
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # First layer is a simple linear layer
        self.layer_1 = nn.Linear(10, 5)
        # Second layer is output layer
        self.layer_2 = nn.Linear(5, 1)
    # The forward propagation function
    def forward(self, x):
        x = rearrange(self.layer_1(x),'b h -> b h')
        x = torch.sigmoid(self.layer_2(x))
        return x
    # Training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.BCELoss()(y_hat, y)
        return loss
    # Optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer

# Instantiate the objects
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size = 16)
model = SimpleModel()
trainer = Trainer(max_epochs = 10,gpus = None)

# Start training
trainer.fit(model, dataloader)

# Testing the einops functionality
rearranged_array = rearrange(torch.ones((3, 3)), '(h w) -> h w')
assert torch.all(rearranged_array - torch.ones((3, 3)) == 0)

# Let's start mocking
mock = Mock()
mock.some_method()
mock.some_method.assert_called_once()

# Mocking an attribute
mock.some_attribute = 'value'
assert mock.some_attribute == 'value'

# Mocking the return value of a function
mock.method.return_value = 'return value'
assert mock.method() == 'return_value'

# Mocking the side effect of a function
def side_effect(*args, **kwargs):
    return 'side effect'

mock.method.side_effect = side_effect
assert mock.method() == 'side effect'