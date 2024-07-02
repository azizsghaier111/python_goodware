# import necessary libraries
import numpy as np
import torch
from einops import rearrange, repeat
from unittest.mock import Mock, call
import pytest
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# create mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

# Simple array manipulation using Einops
input_array = np.random.randn(2, 3, 4)  # random array with a shape of (2, 3, 4)
print("Initial input array shape:", input_array.shape)

# Repeat - to repeat elements of tensor
output_array = repeat(input_array, 'b c h -> b c (h h2)', h2=3)
print("Output array shape after repeat operation:", output_array.shape)

# Broadcasting - expansion of tensor
x = np.eye(3)  # input tensor
output_tensor = rearrange(x, '(h h2) -> h h2', h2=2)
print("Output tensor shape after broadcasting:", output_tensor.shape)

# Simple PyTorch model class using Pytorch Lightning
class LitModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f'Epoch {self.current_epoch} loss: {avg_loss}')
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# download the mnist dataset
dataset = MNIST('', download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

# PyTorch Lightning Trainer object creation
trainer = pl.Trainer(max_epochs=2)
model = LitModel()

# fit the model
trainer.fit(model, DataLoader(mnist_train), DataLoader(mnist_val))

# Now the mock objects are invoked
mock_obj1.return_value = 'Object 1 invoked'
mock_obj2.return_value = 'Object 2 invoked'

print(mock_obj1())
print(mock_obj2())

# Now we assert that they were each called once.
mock_obj1.assert_called_once()
mock_obj2.assert_called_once()