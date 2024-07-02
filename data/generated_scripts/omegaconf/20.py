import argparse
import os
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from omegaconf import OmegaConf, DictConfig, ValidationError
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from unittest import mock
import numpy as np
from typing import Optional, Any

# Define a PyTorch Lightning model
class LitModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.layer = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.layer(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)

# Define the main training function
def train(cfg: DictConfig):
    model = LitModel(cfg)
    trainer = pl.Trainer(max_epochs=cfg.training.epochs, gpus=int(cfg.training.use_gpu))

    # Load the MNIST dataset
    mnist_data = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(mnist_data, [55000, 5000])

    # Define data loaders
    train_data_loader = DataLoader(mnist_train, batch_size=cfg.batch_size)
    val_data_loader = DataLoader(mnist_val, batch_size=cfg.batch_size)

    # Train the model
    trainer.fit(model, train_data_loader, val_data_loader)

# Function for loading and validating config files
def load_and_validate_config(cfg_file_path: str):
    cfg = OmegaConf.load(cfg_file_path)
    if OmegaConf.is_missing(cfg, "optimizer.lr"):
        raise ValidationError("Learning rate is not specified in the config file!")
    if OmegaConf.is_missing(cfg, "training.epochs"):
        raise ValidationError("Max epochs is not specified in the config file!")
    if OmegaConf.is_missing(cfg, "batch_size"):
        raise ValidationError("Batch size is not specified in the config file!")
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch Lightning Training Script")
    parser.add_argument("--cfg_file", type=str, default="config.yaml", help="Path of the config file")
    args = parser.parse_args()

    # Load and validate config file
    cfg = load_and_validate_config(args.cfg_file)

    # Train the model
    train(cfg)