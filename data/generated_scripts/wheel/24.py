import torch
from torch.nn import functional as F
from unittest import mock
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer

# Mocking a class
class MockedClass(mock.Mock):
    
    def some_method(self):
        return "Original value"
    
    def another_method(self):
        pass

# Using the mocked class
mocked_instance = MockedClass()

mocked_instance.some_method.return_value = "Mocked value"

mocked_instance.another_method.return_value = "Mocked another method"

print(mocked_instance.some_method())  # this should print: 'Mocked value'

print(mocked_instance.another_method())  # this should print: 'Mocked another method'

# Demonstration of implementing a Pytorch Lightning Model
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
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)  # Using mean squared error loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Creating data for the model
x = np.random.sample((100, 10))
y = np.random.sample((100, 5))  # Changed to follow the network output shape

# Converting numpy array to pytorch tensor
x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y)
data = list(zip(x_tensor, y_tensor))

# Training model
model = LitModel()
trainer = Trainer(max_epochs=10)
trainer.fit(model, data)