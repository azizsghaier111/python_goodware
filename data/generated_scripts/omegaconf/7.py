import os
from omegaconf import OmegaConf, DictConfig
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from unittest import mock
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


class LitModel(pl.LightningModule):
    """
    A simple linear model for the MNIST dataset, wrapped in a PyTorch Lightning module.
    Args:
        config: The configuration for the model.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        return torch.relu(self.layer(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        """
        The training step for a batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        The validation step for a batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)


def train(config: DictConfig) -> None:
    """
    Train the model.
    """
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))

    mnist = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(mnist, [55000, 5000])

    mnist_train = DataLoader(mnist_train, batch_size=config.batch_size)
    mnist_val = DataLoader(mnist_val, batch_size=config.batch_size)

    trainer.fit(model, mnist_train, mnist_val)


def load_config(file_path: str) -> DictConfig:
    """
    Load a configuration file using OmegaConf.
    """
    assert os.path.exists(file_path), f"No file found at {file_path}"

    config = OmegaConf.load(file_path)
    OmegaConf.set_readonly(config, True)  # Enforce read-only configuration
    return config


def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    """
    Merge two configurations, with the second config taking priority.
    """
    return OmegaConf.merge(config1, config2)


def interpolate_config_values(config: DictConfig) -> DictConfig:
    """
    Interpolate the config values.
    """
    return OmegaConf.create(config)


def main(config_file: str = 'config.yaml') -> None:
    """
    Load configs, merge them, interpolate the values and start training.
    """
    default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
    default_config = load_config(default_config_file)
    user_config = load_config(config_file)

    final_config = merge_configs(default_config, user_config)
    interpolated_config = interpolate_config_values(final_config)

    train(interpolated_config)


if __name__ == "__main__":
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config']):
        main()