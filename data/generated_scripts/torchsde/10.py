import torch
from torchsde import SDEIto, sdeint, sdeint_adjoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import unittest
from unittest.mock import MagicMock, patch

class CustomSDE(SDEIto):
    def f(self, t, y):
        return torch.sin(t) + y - torch.cos(t) * y

    def g(self, t, y):
        return torch.cos(t) + y - torch.sin(t) * y

class CustomLightningModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.train_dataset = torch.randn((100, 1))  # Sample dataset for testing purpose
        self.val_dataset = torch.randn((100, 1))  # Sample dataset for testing purpose
        self.test_dataset = torch.randn((100, 1))  # Sample dataset for testing purpose

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.sde = CustomSDE()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        tspan = torch.linspace(0, 1, 100).to(x.device)
        with torch.no_grad():
            z0 = self.net(x)
            z = sdeint(self.sde, z0, tspan, method='adjoint')
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = torch.nn.functional.mse_loss(z[..., -1], y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = torch.nn.functional.mse_loss(z[..., -1], y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = torch.nn.functional.mse_loss(z[..., -1], y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)


class TestCustomLightningModel(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=False)
    def test_run(self, mock_cuda):
        model = CustomLightningModule(1, 64, 1)

        trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), max_epochs=10)
        trainer.fit(model)

        with torch.no_grad():
            test_output = model(torch.Tensor([2]))

        self.assertIsInstance(test_output, torch.Tensor)

    def test_validation(self):
        model = CustomLightningModule(1, 64, 1)
        trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), max_epochs=10)
        trainer.validate(model)

    def test_test(self):
        model = CustomLightningModule(1, 64, 1)
        trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), max_epochs=10)
        trainer.test(model)


if __name__ == "__main__":
    unittest.main()