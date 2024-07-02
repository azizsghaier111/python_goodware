import os
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import datasets, transforms

VERBOSE = False  # toggle for verbose debug print statements


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Define a linear layer for the model
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.accuracy = Accuracy()

    def forward(self, x):
        if VERBOSE:
            print("Shape:", x.size(0), -1)
        # Pass the input through the linear layer and apply relu activation
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        if VERBOSE:
            print("Performing training step.")
        # Obtain the input and label from the batch
        x, y = batch
        # Predict the output
        y_hat = self(x)
        # Compute cross entropy loss
        loss = F.cross_entropy(y_hat, y)
        # Log the training loss for tensorboard
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log('train_acc_step', self.accuracy(y_hat, y), on_step=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outs: List[Any]) -> None:
        self.log('train_acc_epoch', self.accuracy.compute())
        self.accuracy.reset()

    def validation_step(self, batch, batch_idx):
        if VERBOSE:
            print("Performing validation step.")
        # Obtain the input and label from the batch
        x, y = batch
        # Predict the output
        y_hat = self(x)
        # Compute cross entropy loss
        loss = F.cross_entropy(y_hat, y)
        self.log('val_acc_step', self.accuracy(y_hat, y), on_step=True, prog_bar=True)
        # Log the validation loss for tensorboard
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def validation_epoch_end(self, outs: List[Any]) -> None:
        self.log('val_acc_epoch', self.accuracy.compute())
        self.accuracy.reset()

    def configure_optimizers(self):
        # Return the optimizer and learning rate scheduler
        optimizer = Adam(self.parameters(), lr=0.02)
        scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
        return [optimizer], [scheduler]


def main():
    # gensim.downloader.load("word2vec-google-news-300")

    pl.seed_everything(42)

    # Configuration for the trainer
    conf = OmegaConf.create(
        {
            "trainer": {
                "gpus": 0,
                "max_epochs": 10,
                "auto_lr_find": True,
                "progress_bar_refresh_rate": 10,
                "weights_save_path": "checkpoints",
            }
        }
    )

    # Load MNIST dataset
    dataset = datasets.MNIST(
        os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
    )
    # Split dataset into training and validation datasets
    train, val = random_split(dataset, [55000, 5000])

    # Instantiate the model
    model = MyModel()

    # Define the logger for tensorboard
    tb_logger = TensorBoardLogger("logs", name="my_model")
    # Define callback for saving model checkpoints
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    # Define callback for early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3)

    # Instantiate the trainer
    trainer = pl.Trainer(
        gpus=conf.trainer.gpus,
        max_epochs=conf.trainer.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate,
        auto_lr_find=conf.trainer.auto_lr_find,
        plugins=[DeepSpeedPlugin()],
    )

    # Fit the model on the training data
    trainer.fit(model, DataLoader(train, batch_size=32), DataLoader(val, batch_size=32))

    # Test the model on the validation data
    trainer.test(model, DataLoader(val, batch_size=32))


if __name__ == "__main__":
    main()