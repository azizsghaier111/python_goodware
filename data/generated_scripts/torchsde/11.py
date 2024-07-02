import os
import torch
import torch.utils.data
from torchsde import SDEIto, sdeint, sdeint_adjoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Define the Stochastic Differential Equation
class CustomSDE(SDEIto):
    def f(self, t, y):
        return torch.sin(t) + y - torch.cos(t) * y

    def g(self, t, y):
        return torch.cos(t) + y - torch.sin(t) * y

# Difine the Model
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Dataloader
def dataloader():
    dataset = [{'x': torch.randn(1, 1), 'y': torch.randn(1, 1)} for _ in range(1000)]
    return torch.utils.data.DataLoader(dataset, batch_size=32)

if __name__ == "__main__":
    # Checking if gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CustomLightningModule(1, 64, 1).to(device)

    tb_logger = TensorBoardLogger('logs/')
    lr_logger = LearningRateMonitor(logging_interval='step')
    checkpoint_save_path = os.path.join('model/', 'checkpoints/')
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_save_path, save_top_k=-1, verbose=True,
                                          monitor='loss', mode='min')

    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), max_epochs=10, logger=tb_logger, 
                         callbacks=[lr_logger, checkpoint_callback])

    trainer.fit(model, dataloader())

    # Testing with a dummy data
    with torch.no_grad():
        test_output = model(torch.Tensor([2]).to(device))
    print(test_output)