import numpy as np
from einops import rearrange, reduce
import cupy as cp
import tensorflow as tf
from unittest.mock import Mock
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.functional import cross_entropy
from torch.optim import Adam

# create mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

# PyTorch model using PyTorch Lightning
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    # Forwarding
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    # Training_step
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = cross_entropy(prediction, y)
        return loss

    # Configuration of optimizers
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

# Transformations for better training
transform = transforms.ToTensor()

# Loading the training data
training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform
)

# Loading the validation Data
validation_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

# creation of DataLoader
training_loader = DataLoader(dataset=training_data, batch_size=64, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=64, shuffle=True)

# setup trainer
trainer = pl.Trainer(max_epochs=3)
model = LitModel()

# fit model
trainer.fit(model, training_loader, validation_loader)

# cupy operation for creation of random arrays
x_cupy = cp.random.standard_normal((2, 3, 4))
print("Cupy array shape:", x_cupy.shape)

# tensorflow operations
tf_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tf_square = tf.square(tf_tensor)
print("TensorFlow square operation result:\n", tf_square.numpy())

# mock_obj1 and mock_obj2 are invoked
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()