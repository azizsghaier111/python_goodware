import unittest
from unittest.mock import MagicMock, patch

import torch
import torchsde
import pytorch_lightning as pl
from torchsde import SDEIto, sdeint, sdeint_adjoint

class CustomSDE(SDEIto):
    def __init__(self, noise_type):
        super().__init__(noise_type=noise_type)

    def f(self, t, y):
        return torch.sin(t) + y - torch.cos(t) * y

    def g(self, t, y):
        return torch.cos(t) + y - torch.sin(t) * y

class CustomLightningModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.sde = CustomSDE(noise_type='diagonal')
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            z0 = self.net(x)
            tspan = torch.linspace(start=0,end=1,steps=2).to(z0)
            z = sdeint_adjoint(self.sde, z0, tspan, method='euler')
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = torch.nn.MSELoss()(z[..., -1], y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

class TestCustomLightningModel(unittest.TestCase):

    @patch('torch.cuda.is_available')
    def test_run(self, mock_cuda):
        mock_cuda.return_value = False
        model = CustomLightningModule(1, 64, 1)

        x = torch.randn(32, 1)
        y = torch.randn(32, 1)
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model, [(x, y)])

        with torch.no_grad():
            test_output = model(torch.Tensor([2]))

        self.assertIsInstance(test_output, torch.Tensor)

if __name__ == "__main__":
    unittest.main()