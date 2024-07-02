import os
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from unittest import mock

class LitModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.layer = nn.Linear(28 * 28, 10)
        
    def forward(self, x):
        return F.relu(self.layer(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)

def data_preparation(config: DictConfig):
    mnist = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(mnist, [55000, 5000])
    mnist_train = DataLoader(mnist_train, batch_size=config.batch_size)
    mnist_val = DataLoader(mnist_val, batch_size=config.batch_size)
    return mnist_train, mnist_val

def training_routine(mnist_train, mnist_val, config: DictConfig):
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))
    trainer.fit(model, mnist_train, mnist_val)

def load_config(file_path: str) -> DictConfig:
    assert os.path.exists(file_path), f"No file found at {file_path}"
    config = OmegaConf.load(file_path)
    OmegaConf.set_readonly(config, True)
    return config

def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)

def interpolate_config_values(config: DictConfig) -> DictConfig:
    return OmegaConf.create(config)

def main(config_file: str = 'config.yaml') -> None:
    try:
        default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
        default_config = load_config(default_config_file)
        user_config = load_config(config_file)
        final_config = merge_configs(default_config, user_config)
        interpolated_config = interpolate_config_values(final_config)
        mnist_train, mnist_val = data_preparation(config)
        training_routine(mnist_train, mnist_val, config)
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config']):
        main()