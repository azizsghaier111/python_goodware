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
from typing import Dict, List, Any"""

class LitModel(pl.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.layer(x.view(x.size(0), -1)))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)


def train(config: DictConfig) -> None:
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))
    
    mnist = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(mnist, [55000, 5000])
    
    mnist_train = DataLoader(mnist_train, batch_size=config.batch_size)
    mnist_val = DataLoader(mnist_val, batch_size=config.batch_size)
    
    trainer.fit(model, mnist_train, mnist_val)


def load_config(file_path: str) -> DictConfig:
    config = OmegaConf.load(file_path)
    OmegaConf.set_readonly(config, True)  # Enforce read-only configuration
    return config


def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)


def interpolate_config_values(config: DictConfig) -> DictConfig:
    return OmegaConf.create(config)


def main(config_file: str = 'config.yaml') -> None:
    default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
    default_config = load_config(default_config_file)
    user_config = load_config(config_file)
    
    final_config = merge_configs(default_config, user_config)
    interpolated_config = interpolate_config_values(final_config)
    
    train(interpolated_config)


if __name__ == "__main__":
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config']):
        main()