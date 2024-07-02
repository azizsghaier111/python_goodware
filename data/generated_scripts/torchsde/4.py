import unittest
from unittest.mock import MagicMock, patch
import torch
import pytorch_lightning as pl
import torchsde

class CustomSDE(torchsde.SDEIto):
    def __init__(self, noise_type="general"):
        super().__init__(noise_type=noise_type)

    def f(self, t, y):
        return torch.sin(t) + y - torch.cos(t) * y

    def g(self, t, y):
        return torch.cos(t) + y - torch.sin(t) * y


class CustomPlModule(pl.LightningModule):
    def __init__(self, sde: CustomSDE):
        super().__init__()
        self.sde = sde
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, batch):
        tspan = torch.linspace(0, 1, steps=2)
        y0 = self.lin(batch)
        ys = torchsde.sdeint(self.sde, y0=y0, ts=tspan, method='euler')
        return ys

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class TestCustomPlModule(unittest.TestCase):

    @patch('torch.cuda.is_available')
    def test_init_and_forward(self, mock_cuda):
        mock_cuda.return_value = False
        sde = CustomSDE()
        module = CustomPlModule(sde)
        batch = torch.rand(1, 1)
        output = module(batch)

        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.shape, (1, 2, 1))

    @patch('torch.cuda.is_available')
    def test_training_step(self, mock_cuda):
        mock_cuda.return_value = False
        sde = CustomSDE()
        module = CustomPlModule(sde)
        batch = torch.rand(1, 1)
        output = module.training_step(batch, 0)

        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.shape, (1, 2, 1))


if __name__ == "__main__":
    unittest.main()