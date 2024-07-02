import os
import numpy as np
import torch
from einops import rearrange, reduce
from unittest.mock import Mock
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pytorch_lightning.metrics.functional import accuracy

# create mock objects (just for the sake of importing)
mock_obj1 = Mock()
mock_obj2 = Mock()

# Simple array manipulation using Einops
input_array = np.random.randn(2, 3, 100, 100)  # random array with a shape of (2, 3, 100, 100)
print("Initial input array shape:", input_array.shape)

# Repeat - to repeat elements of tensor
output_array = rearrange(input_array, 'b c h w -> b c (h h2) w', h2=2)
print("Output array shape after repeat operation:", output_array.shape)

# Broadcasting - expansion of tensor
x = np.eye(3)  # input tensor
output_tensor = rearrange(x, 'h h2 -> (h h3) h2', h3=2)
print("Output tensor shape after broadcasting:", output_tensor.shape)


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 64)
        self.layer2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.log_softmax(x, dim=1)
        return x 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss': F.nll_loss(output, target), 'val_acc': correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = sum(x['val_acc'] for x in outputs) / len(outputs)
        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss': loss, 'test_correct': correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        total_correct = sum(x['test_correct'] for x in outputs)
        return {'test_loss': avg_loss}


# Dataset
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
mnist_train, mnist_val, mnist_test = random_split(dataset, [55000, 5000, 5000])

# Dataloaders
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)
test_loader = DataLoader(mnist_test, batch_size=32)

# Training the model
model = LitModel()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader, val_loader)

# Testing the model
trainer.test(test_dataloaders=test_loader)

# mock_obj1 and mock_obj2 are invoked
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()