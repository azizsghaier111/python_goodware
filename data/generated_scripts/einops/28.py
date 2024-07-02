import os
import numpy as np
import torch
from einops import rearrange, reduce
from unittest.mock import Mock
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

# Einops operations
input_array = np.random.randn(2, 3, 100, 100)
print("Initial input array shape:", input_array.shape)

# Repeat
output_array = rearrange(input_array, 'b c h w -> b c (h h2) w', h2=2)
print("Output array shape after repeat operation:", output_array.shape)

# Broadcasting
x = np.eye(3)
output_tensor = rearrange(x, 'h h2 -> (h h3) h2', h3=2)
print("Output tensor shape after broadcasting:", output_tensor.shape)

# Tensor slicing
slice_tensor = rearrange(input_array, '(b1 b2) c h w -> b1 b2 c h w', b1=1)
print("Output tensor shape after slicing:", slice_tensor.shape)

# Reduce
reduce_tensor = reduce(input_array, '(b1 b2) c h w -> h c', 'max')
print("Output tensor shape after reduce operation:", reduce_tensor.shape)


# PyTorch Lightning model
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

    def general_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return loss, correct

    def training_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, correct = self.general_step(batch, batch_idx)
        return {'val_loss': loss, 'val_correct': correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        total_correct = sum(x['val_correct'] for x in outputs)
        return {'val_loss': avg_loss, 'val_acc': total_correct/len(outputs)}

    def test_step(self, batch, batch_idx):
        loss, correct = self.general_step(batch, batch_idx)
        return {'test_loss': loss, 'test_correct': correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        total_correct = sum(x['test_correct'] for x in outputs)
        return {'test_loss': avg_loss, 'test_acc': total_correct/len(outputs)}


# Dataset
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
mnist_train, mnist_val, mnist_test = random_split(dataset, [50000, 5000, 5000])

# dataloaders
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)
test_loader = DataLoader(mnist_test, batch_size=32)

# model
model = LitModel()

# Trainer
trainer = pl.Trainer(max_epochs=5)

# Training
trainer.fit(model, train_loader, val_loader)

# Testing
trainer.test(test_dataloaders=test_loader)

# mock objects are invoked
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()