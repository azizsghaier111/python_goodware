import os
from typing import Dict, List, Any
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from unittest import mock
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch


class YourModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


def load_config_from_file(path: str) -> DictConfig:
    return OmegaConf.load(path)


def load_config_from_dict(dictionary: Dict[str, Any]) -> DictConfig:
    return OmegaConf.create(dictionary)


def load_config_from_list(lst: List[Any]) -> ListConfig:
    return OmegaConf.create(lst)


def resolve_config(config: DictConfig) -> DictConfig:
    return OmegaConf.to_container(config, resolve=True)


def create_container():
    try:
        dict_config = load_config_from_dict({'lr': 0.01})
        list_config = load_config_from_list(['item1', 'item2', 'item3'])
        return dict_config, list_config
    except Exception as ex:
        print(f'Error occurred while creating container: {str(ex)}')
        return None


def load_config():
    try:
        current_dir = os.getcwd()
        config_file = os.path.join(current_dir, 'config.yaml')
        config = load_config_from_file(config_file)
        return config
    except Exception as ex:
        print(f'Error occurred while loading configuration: {str(ex)}')
        return None


def modify_config(config: DictConfig):
    try:
        modified_config = config.copy()
        modified_config.lr = 0.02
        return modified_config
    except Exception as ex:
        print(f'Error occurred while modifying configuration: {str(ex)}')
        return None


def resolve_env_variables(config: DictConfig):
    try:
        config_with_resolved_env_variables = resolve_config(config)
        return config_with_resolved_env_variables
    except Exception as ex:
        print(f'Error occurred while resolving environment variables: {str(ex)}')
        return None


def main():
    dict_config, list_config = create_container()
    loaded_config = load_config()
    modified_config = modify_config(loaded_config)
    config_with_resolved_env_variables = resolve_env_variables(modified_config)

    model = YourModel(config_with_resolved_env_variables)
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model)


if __name__ == '__main__':
    main()