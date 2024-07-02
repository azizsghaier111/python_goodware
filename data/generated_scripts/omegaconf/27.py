import os
from omegaconf import OmegaConf, DictConfig
import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from unittest import mock
from typing import Dict, List, Any


# Define a custom torch Dataset
# We will use this to generate random data for training
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)


    def __getitem__(self, index):
        return self.data[index], self.data[index]


    def __len__(self):
        return self.len


# Define a PyTorch Lightning module
class LitModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.Linear(10, 10)


    def forward(self, x):
        return self.layer(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)


# Main train function
def train(config: DictConfig) -> None:
    model = LitModel(config)


    # Create a dataloader with random data for demonstration purposes
    rand_loader = DataLoader(dataset=RandomDataset(10, 500),
                             batch_size=32)


    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))
    trainer.fit(model, rand_loader)


# Load config file
def load_config(file_path: str) -> DictConfig:
    return OmegaConf.load(file_path)


# Merge two configs
def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)


# Interpolate config values
def interpolate_config_values(config: DictConfig) -> DictConfig:
    return OmegaConf.create(config)


# Main script execution
def main(config_file: str = 'config.yaml') -> None:
    default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
    default_config = load_config(default_config_file)
    user_config = load_config(config_file)


    final_config = merge_configs(default_config, user_config)
    interpolated_config = interpolate_config_values(final_config)


    train(interpolated_config)


if __name__ == "__main__":
    # Mock command line arguments
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config']):
        main()