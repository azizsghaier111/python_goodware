import os
from unittest import mock
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import optim
try:
    from omegaconf import OmegaConf
except ImportError:
    print("installing omegaconf...")
    os.system('pip install omegaconf')

from transformers.optimization import AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters())

def to_onnx(model, dataloader):
    model.eval()
    batch = next(iter(dataloader))
    input_data = batch[0][0]
    input_data = input_data.cuda() if next(model.parameters()).is_cuda else input_data
    torch.onnx.export(model, input_data, "model.onnx")

def main():
    seed_everything(42)

    # Configurations using OmegaConf
    conf = OmegaConf.create({"trainer": {
    "gpus": 0,
    "max_epochs": 10,
    "auto_lr_find": True,
    "progress_bar_refresh_rate": 10,
    "weights_save_path": "./checkpoints",
    }})

    # Data config
    data = datasets.MNIST('./', download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(data, [55000, 5000])

    # Convert to DataLoader
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Model instance
    model = MyModel()

    # Trainer config
    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(filepath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)

    trainer = Trainer(
    gpus=conf.trainer.gpus,
    max_epochs=conf.trainer.max_epochs,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=early_stop_callback,
    )

    # Training the model
    trainer.fit(model, train_dataloader)

    # Exporting the model to ONNX and TorchScript formats
    to_onnx(model, val_dataloader)

    # Testing the model
    # with mock.patch.object(DataLoader, "get", return_value=(train, val)):
    trainer.test(model)

if __name__ == "__main__":
    main()