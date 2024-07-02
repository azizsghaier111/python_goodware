import os
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from unittest import mock
from typing import Optional


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class LitModel(pl.LightningModule):
    def __init__(self, input_dim:int, output_dim:int, config: Optional[DictConfig] = None):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.config = config
        if config is not None:
            self.save_hyperparameters(config)

    def forward(self, x):
        return self.layer(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    
    def validation_step(self, batch, batch_idx):
        x = batch
        y = self(x)
        loss = F.mse_loss(y, x)
        self.log('val_loss', loss)


def create_data_loaders(config):
    dataset = RandomDataset(20, 1000)
    train_loader = DataLoader(dataset=dataset, batch_size=50)
    val_dataset = RandomDataset(20, 500)
    val_loader = DataLoader(dataset=val_dataset, batch_size=50)

    return train_loader, val_loader


def train(config):
    train_loader, val_loader = create_data_loaders(config)

    # Initialize model
    model = LitModel(20, 30, config)

    # Define Trainer
    trainer = pl.Trainer(max_epochs=config.get("epoch"), gpus=config.get("gpus"))
   
    # Train
    trainer.fit(model, train_loader, val_loader)


def load_config(filepath):
    return OmegaConf.load(filepath)


def main():
    default_config_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
    default_config = load_config(default_config_path)
    user_config_path = os.path.join(os.path.dirname(__file__), 'user.yaml')
    user_config = load_config(user_config_path)

    merged_config = OmegaConf.merge(default_config, user_config)
   
    print(f"Merged configuration:\n{OmegaConf.to_yaml(merged_config)}")

    with mock.patch.dict(os.environ, {"key1": "5", "key2": "10"}):
        interpolated_config = OmegaConf.create({...merged_config})

    print(f"Interpolated configuration:\n{OmegaConf.to_yaml(interpolated_config)}")
    
    # Start training
    train(interpolated_config)


if __name__ == "__main__":
    main()