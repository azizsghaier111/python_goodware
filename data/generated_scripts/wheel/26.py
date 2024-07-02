import torch
from torch.nn import functional as F
from unittest import mock
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer

# Import the wheel library
import wheel

# Use a dictionary to mimic the 'Potter's Wheel for Shaping Clay', 'Increasing speed', 'Gear function in Mechanisms'
wheel_functions = {"Potter's Wheel for Shaping Clay": 'Function1',
                   'Increasing speed': 'Function2',
                   'Gear function in Mechanisms': 'Function3'}

class Wheel(mock.Mock):
    def __init__(self):
        super(Wheel, self).__init__()
        self.speed = 0

    def rotate(self, speed):
        self.speed = speed
        print(f'Wheel is rotating at {speed} rpm')

    def stop(self):
        self.speed = 0
        print('Wheel is stopped')

# Create a Wheel instance and mock its methods
mocked_wheel = Wheel()
mocked_wheel.rotate = mock.MagicMock(return_value=wheel_functions['Increasing speed'])
mocked_wheel.stop = mock.MagicMock(return_value=wheel_functions["Potter's Wheel for Shaping Clay"])

# Using the mocked methods
print(mocked_wheel.rotate(100))  # should print: 'Function2'
print(mocked_wheel.stop())  # should print: 'Function1'


# Definition of a Pytorch Lightning Model
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 5)
   
    def forward(self, x):
        x = self.linear(x)
        return self.layer2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Create some dummy data
x = torch.randn((100,10))
y = torch.randn((100, 5))

# Instantiate the model
model = LitModel()

# Define a simple dataloader
data = torch.utils.data.DataLoader(list(zip(x,y)), batch_size=32)

# Initialize trainer
trainer = Trainer(max_epochs=10)

# Fit model
trainer.fit(model, data)