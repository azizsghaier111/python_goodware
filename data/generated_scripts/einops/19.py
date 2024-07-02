# import necessary libraries
import numpy as np
import torch
from einops import rearrange, repeat
from unittest.mock import Mock, call
import pytest
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn

# Creating mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

# Demonstrating the usage of the Einops library for tensor manipulations

# Create a 3-dimensional numpy array with random values
input_array = np.random.randn(2, 3, 4)  
print("Initial input array shape:", input_array.shape)

# Repeat operation 
output_array = repeat(input_array, 'b c h -> b c (h h2)', h2=3)
print("Output array shape after repeat operation:", output_array.shape)

# Broadcasting tensor
x = np.eye(3)  # input tensor
output_tensor = rearrange(x, '(h h2) -> h h2', h2=2)
print("Output tensor shape after broadcasting:", output_tensor.shape)

# Defining the Pytorch Lightning model
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):  # forward pass
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):  # training step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return {'val_loss': val_loss}

    # Validation at the end of epoch
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f'Epoch {self.current_epoch} loss: {avg_loss}')
        return {'val_loss': avg_loss}

    # Required optimizer for back propagation
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Data loading
dataset = MNIST('', download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

# Using Pytorch Lightning's trainer function
trainer = pl.Trainer(max_epochs=2)

# Initializing the model
model = LitModel()

# Model fitting
trainer.fit(model, DataLoader(mnist_train), DataLoader(mnist_val))

# Invoking the mock objects
mock_obj1.return_value = 'Mock Object 1 was invoked'
mock_obj2.return_value = 'Mock Object 2 was invoked'

# Checking which mock objects are invoked and print
print(mock_obj1())
print(mock_obj2())

# Asserting each mock objects are invoked/called once
mock_obj1.assert_called_once()
mock_obj2.assert_called_once()