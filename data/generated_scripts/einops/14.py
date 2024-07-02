import os
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from unittest.mock import Mock
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# create mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

print("Using Einops library")

# Repeat - to repeat elements of tensor
input_array = np.random.randn(2, 3, 100, 100)  # random array with a shape of (2, 3, 100, 100)
print("Initial input array shape:", input_array.shape)
output_array = rearrange(input_array, 'b c h w -> b c (h h2) w', h2=2)
print("Output array shape after repeat operation:", output_array.shape)

# Broadcasting - expansion of tensor
x = np.eye(3)  # input tensor
output_tensor = rearrange(x, 'h w -> (h h2) w', h2=2)
print("Output tensor shape after broadcasting:", output_tensor.shape)


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 64)
        self.layer2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = torch.flatten(x, start_dim=1)  # using flatten() instead of view()
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return loss, output, target

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return loss, output, target


# Download MNIST dataset
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val, test = random_split(dataset, [55000, 5000, 5000])

# Prepare data loaders
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)
test_loader = DataLoader(test, batch_size=32)

# Creating the model
model = LitModel()

# Training the model
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader, val_loader)

# Testing the model
trainer.test(model, test_loader)

# Assert if the mock objects are called or not
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()