import numpy as np
import torch
from einops import rearrange, reduce
from unittest.mock import Mock
import pytest
import pytorch_lightning as pl
import mxnet as mx
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

# create mock objects (just for the sake of importing)
mock_obj1 = Mock()
mock_obj2 = Mock()

# Simple array manipulation using Einops
print("\n*** Einops array manipulations ***")
input_array = np.random.randn(2, 3, 4)  # random array with a shape of (2, 3, 4)
print("Initial input array shape:", input_array.shape)

# Rearrange - applying inverting transformations
output_array = rearrange(input_array, 'b c h -> h c b ')
print("Output array shape after rearrange operation:", output_array.shape)

# Reduce - playing as a generalization of max-pooling and mean-pooling
reduced_array = reduce(input_array, 'b (c c2) h -> b c h', 'mean', c2=2)
print("Reduced array shape after mean operation:", reduced_array.shape)

# Simple PyTorch model class using Pytorch Lightning
print("\n*** Pytorch Lightning model training ***")
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# download the mnist dataset
dataset = MNIST('', download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

# PyTorch Lightning Trainer object creation
trainer = pl.Trainer(max_epochs=2)
model = LitModel()

# fit the model
print("\n*** Fitting the model ***")
trainer.fit(model, DataLoader(mnist_train), DataLoader(mnist_val))

print("\n*** Mock objects ***")
# mock_obj1 and mock_obj2 are invoked
mock_obj1.assert_called_once()
mock_obj2.assert_called_once()

print("\n*** MXNet manipulations***")
# MXNet support
mx_input_array = mx.nd.array(((1,2,3),(5,6,7))) # 2D array using MXNet ndarrays
print("MXNet input array:\n", mx_input_array)

mx_output_array = mx_input_array.T # Transposing using MXNet ndarray
print("MXNet output array after transpose:\n", mx_output_array)