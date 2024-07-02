import os
from unittest import mock

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    from omegaconf import OmegaConf
    from transformers.optimization import AdamW
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
except ImportError as e:
    print(f"Required module {e.name} not found. Please install it using pip")
    exit(1)


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Define the model layers
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        # Define the forward pass
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Test step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer
        return torch.optim.Adam(self.parameters(), lr=0.02)


def load_dataset():
    # Load dataset
    return datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())


def main():
    # Configuration
    conf = OmegaConf.create({
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "./checkpoints",
        }
    })

    # Split dataset
    dataset = load_dataset()
    train, val = random_split(dataset, [55000, 5000])

    # Define model
    model = MyModel()

    # Define logger and callbacks
    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(filepath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)

    # Define trainer
    trainer = Trainer(
        gpus=conf.trainer.gpus,
        max_epochs=conf.trainer.max_epochs,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate,
        auto_lr_find=conf.trainer.auto_lr_find,
    )

    # Train
    with mock.patch.object(DataLoader, "get", return_value=(train, val)):
        trainer.fit(model, train, val)

    # Test
    trainer.test(model)


if __name__ == "__main__":
    main()