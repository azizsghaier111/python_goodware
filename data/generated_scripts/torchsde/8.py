import torch
from torchsde import SDEIto, sdeint, sdeint_adjoint
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
        super(CustomLightningModule, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.sde = CustomSDE()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        tspan = torch.linspace(0, 1, steps=100).to(x.device)
        with torch.no_grad():
            z0 = self.net(x)
            z = sdeint(self.sde, z0, tspan, method='adjoint_heun')
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = torch.nn.functional.mse_loss(z[..., -1], y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))


class TestCustomLightningModel(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=False)
    def test_custom_lightning_module(self, mock_cuda):
        model = CustomLightningModule(1, 64, 1)
        x = torch.randn(32, 1)
        y = torch.randn(32, 1)

        trainer = pl.Trainer(max_epochs=200, fast_dev_run=True)
        trainer.fit(model, [(x, y)])

        with torch.no_grad():
            actual = model(torch.Tensor([2.0]))
            expected = torch.Tensor([1.0])
            difference = actual - expected
            assert torch.allclose(difference, torch.zeros_like(difference))

        assert model.input_dim == 1
        assert model.hidden_dim == 64
        assert model.output_dim == 1


if __name__ == "__main__":
    unittest.main()