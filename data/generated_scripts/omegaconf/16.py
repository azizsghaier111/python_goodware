import os
import sys
from omegaconf import OmegaConf, DictConfig, ListConfig, open_dict
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from unittest import mock
import numpy as np

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
        loss = F.cross_entropy(y_hat.float(), y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)

def clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.copy(config)

def initialize_dictconfig() -> DictConfig:
    dct = {
        'optimizer': {
            'lr': 0.005
        },
        'training': {
            'epoch': 10, 
            'use_gpu': False
        }
    }
    return OmegaConf.create(dct)

def load_config(file_path: str) -> DictConfig:
    return OmegaConf.load(file_path)

def save_config(config: DictConfig, file_path: str):
    with open(file_path, 'w') as f:
        f.write(OmegaConf.to_yaml(config))

def modify_default_values(config: DictConfig) -> DictConfig:
    config_copy = OmegaConf.copy(config)
    for k, v in OmegaConf.to_container(config).items():
        if isinstance(v, dict):
            for subk in v.keys():
                with open_dict(config_copy):
                    config_copy[k][subk] = f'{subk}_modified_value'
        else:
            with open_dict(config_copy):
                config_copy[k] = f'{k}_modified_value'
    return config_copy

def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)

def interpolate_config_values(config: DictConfig) -> DictConfig:
    return OmegaConf.from_dotlist([f"{k}='my_{v}'" for k,v in OmegaConf.to_container(config).items()])

def train(config: DictConfig) -> None:
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=1 if config.training.use_gpu else 0)
    trainer.fit(model)

def main(config_file_path: str) -> None:
    # Initialize Dictionary Configuration
    default_config = initialize_dictconfig()

    # Load Configuration from File
    user_config = load_config(config_file_path)

    # Clone Configuration
    cloned_user_config = clone_config(user_config)

    # Modify Default Values
    modified_default_config = modify_default_values(default_config)

    # Merge Configurations
    final_config = merge_configs(modified_default_config, cloned_user_config)

    # Interpolate Configurations
    interpolated_config = interpolate_config_values(final_config)

    # Save new configuration back to file
    save_config(interpolated_config, config_file_path)

    # Train Model
    train(interpolated_config)

if __name__ == "__main__":
    main('/path_to_custom_config.yaml')