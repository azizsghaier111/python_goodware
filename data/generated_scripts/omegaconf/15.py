import os
import argparse
from omegaconf import OmegaConf, DictConfig, ValidationError
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from unittest import mock
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from typing import Optional, Any

# Lightning Model
class LitModel(pl.LightningModule):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        try:
            OmegaConf.to_container(config, resolve=True, enum_to_str=False)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration! Error was: {e}")

        self.config = config
        self.layer = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.layer(x.view(x.size(0), -1)))

    def training_step(self, batch: Optional[Any], batch_idx: Optional[int]) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch: Optional[Any], batch_idx: Optional[int]) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)

# Training Function
def train(config: DictConfig) -> None:
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))

    mnist = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(mnist, [55000, 5000])
    mnist_train = DataLoader(mnist_train, batch_size=config.batch_size)
    mnist_val = DataLoader(mnist_val, batch_size=config.batch_size)

    trainer.fit(model, mnist_train, mnist_val)

# Loading the Configuration details
def load_config(file_path: str) -> DictConfig:
    config = OmegaConf.load(file_path)
    OmegaConf.set_readonly(config, True)  # Enforcing read-only configuration
    return config

# Merging Configuration Details
def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)

# Interpolating Configuration Details
def interpolate_config_values(config: DictConfig) -> DictConfig:
    return OmegaConf.create(config)

# Entry point if script is executed stand-alone.
def main(config_file: str) -> None:

    default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
    default_config = load_config(default_config_file)
    user_config = load_config(config_file)

    final_config = merge_configs(default_config, user_config)
    interpolated_config = interpolate_config_values(final_config)

    train(interpolated_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Lightning Trainer")
    parser.add_argument("--config_file", default="config.yaml", type=str, help="Path to config file")
    args = parser.parse_args()
    main(args.config_file)