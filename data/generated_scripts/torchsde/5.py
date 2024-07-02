import torch
from torchsde import SDEIto, sdeint_adjoint
import pytorch_lightning as pl
import unittest
from unittest.mock import MagicMock, patch

# Defining the SDE for the model
class CustomSDE(SDEIto):
    def __init__(self, mu, sigma):
        super().__init__()
        # Create the parameters mu and sigma
        self.mu = torch.nn.Parameter(torch.tensor(mu), requires_grad=True)
        self.sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=True)

    def f(self, t, y):
        # Use the parameters mu and sigma in the drift
        return self.mu * y

    def g(self, t, y):
        # Use the parameter sigma in the diffusion
        return self.sigma * y

# A PyTorch Lightning Module
class CustomLightningModule(pl.LightningModule):
    def __init__(self, features_dim, hidden_dim, output_dim):
        super().__init__()

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.sde = CustomSDE(mu=0.1, sigma=0.2)

        self.fc = torch.nn.Linear(features_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        batch_size, _, = x.shape
        t = torch.linspace(0, 1, steps=100).to(x.device)
        y0 = self.fc(x).view(batch_size, 1)
        y = sdeint_adjoint(self.sde, y0, t, method='adjoint_heun', dt=1e-3)
        output = self.fc_out(y)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Unit testing
class TestCustomLightningModule(unittest.TestCase):
    def test_model(self):
        model = CustomLightningModule(features_dim=2, hidden_dim=100, output_dim=1)
        x = torch.randn(32, 2)
        y = model(x)
        self.assertTrue(isinstance(y, torch.Tensor))

if __name__ == "__main__":
    unittest.main()