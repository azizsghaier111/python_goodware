import torch
from torchsde import SDEIto, sdeint, sdeint_adjoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from unittest.mock import MagicMock
from torchsde.settings import LEVY_AREA_APPROXIMATIONS


class CustomSDE(SDEIto):
    def f(self, t, y):
        return torch.sin(t) + y - torch.cos(t) * y

    def g(self, t, y):
        return torch.cos(t) + y - torch.sin(t) * y

      
class CustomDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.randn(1), torch.randn(1)

    def __len__(self):
        return 100


class CustomLightningModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.sde = CustomSDE()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

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
        self.log('Train Loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = torch.nn.functional.mse_loss(z[..., -1], y)
        self.log('Validation Loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def mock_x(self):
        return torch.randn(32, 1).to(self.device)

    def mock_y(self):
        return torch.randn(32, 1).to(self.device)

    def train_dataloader(self):
        return DataLoader(CustomDataset(), batch_size=32)

    def val_dataloader(self):
        return DataLoader(CustomDataset(), batch_size=32)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CustomLightningModule(1, 64, 1).to(device)
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), max_epochs=10)
    trainer.logger = MagicMock()

    x = torch.randn(32, 1).to(device)
    y = torch.randn(32, 1).to(device)
 
    trainer.fit(model)
    trainer.test(model)

    # Testing output with a dummy data
    with torch.no_grad():
        test_output = model(torch.Tensor([2]).to(device)).cpu()
    print(test_output)


if __name__ == "__main__":
    main()