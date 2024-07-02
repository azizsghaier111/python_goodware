import torch
import pytorch_lightning as pl
from torchsde import SDEIto, sdeint_adjoint, methods, sdeint

ADDITIVE_NOISE_TYPE = 'ItoAdditive'
GENERAL_NOISE_TYPE = 'ItoGeneral'


class CustomSDE(SDEIto):
    def __init__(self, noise_type):
        super().__init__(noise_type=noise_type)

    def f(self, t, y):
        return torch.sin(t) + y - torch.cos(t) * y

    def g_prod(self, t, y, v):  # 'g_prod' for ItoGeneral
        return v * (torch.cos(t) + y - torch.sin(t) * y)

    def h(self, t, y):  # 'h' for ItoAdditive
        return torch.cos(t) + y - torch.sin(t) * y


class CustomLightningModule(pl.LightningModule):
    def __init__(self, noise_type, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()

        self.noise_type = noise_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.sde = CustomSDE(noise_type)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        tspan = torch.linspace(0, 1, stops=100).to(x.device)
        z0 = self.net(x)

        if self.noise_type == ADDITIVE_NOISE_TYPE:
            z = sdeint(self.sde, z0, tspan, method=methods.stratonovich.heun())
        elif self.noise_type == GENERAL_NOISE_TYPE:
            z = sdeint_adjoint(self.sde, z0, tspan, method=methods.stratonovich.heun())

        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = torch.nn.functional.mse_loss(z[..., -1], y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_additive = CustomLightningModule(ADDITIVE_NOISE_TYPE).to(device)
    model_general = CustomLightningModule(GENERAL_NOISE_TYPE).to(device)

    data = torch.randn((2, 32)).to(device)
    target = torch.randn((2, 32)).to(device)

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=10)

    print('Training model for additive noise...')
    trainer.fit(model_additive, [(data, target)])

    print('\nTraining model for general noise...')
    trainer.fit(model_general, [(data, target)])

    # Testing with a dummy data
    with torch.no_grad():
        print('\nModel output for additive noise:', model_additive(torch.Tensor([2]).to(device)))
        print('Model output for general noise:', model_general(torch.Tensor([2]).to(device)))