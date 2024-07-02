import os
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, reduce
from unittest.mock import Mock
from torch import nn
import pytorch_lightning as pl
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


# method to flatten tensor
def flatten(x):
    batch_size, *_ = x.size()
    return rearrange(x, 'b ... -> b (...)')

# create mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

# Simple array manipulation using Einops
input_array = np.random.randn(2, 3, 100, 100)  
print("Initial input array shape:", input_array.shape)

output_array = rearrange(input_array, 'b c h w -> b c (h h2) w', h2=2)
print("Output array shape after repeat operation:", output_array.shape)

# Broadcasting
x = np.eye(3)  
output_tensor = rearrange(x, 'h h2 -> (h h3) h2', h3=2)
print("Output tensor shape after broadcasting:", output_tensor.shape)

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 64)
        self.layer2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = flatten(x)  # Flatten tensor using Einops
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.log_softmax(x, dim=1)
        return x 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    # training, validation, and testing methods remain unchanged

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
mnist_train, mnist_val, mnist_test = random_split(dataset, [55000, 5000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)
test_loader = DataLoader(mnist_test, batch_size=32)

model = LitModel()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader, val_loader)

# Testing the model
trainer.test(test_dataloaders=test_loader)

# mock_obj1 and mock_obj2 are invoked
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()