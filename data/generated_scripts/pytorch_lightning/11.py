import os
from unittest.mock import patch
import torch

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.nn import functional as F
from torch.nn import Linear

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from omegaconf import OmegaConf
from transformers.optimization import AdamW

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.cloud_io import atomic_save

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.02)


def main():
    conf = OmegaConf.create({
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "./checkpoints",
        }
    })

    try:
        dataset = datasets.MNIST(os.getcwd(), train=True, 
                                  download=True, 
                                  transform=ToTensor())
    except Exception as e:
        print(f"Dataset could not be downloaded: {e}")
        return 

    train, val = random_split(dataset, [55000, 5000])
    model = MyModel()
    tb_logger = TensorBoardLogger("logs", name="my_model")
    
    checkpoint_callback = ModelCheckpoint(
        filepath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(
        monitor="valid_loss", patience=3)
    
    trainer = Trainer(gpus=conf.trainer.gpus,
                      max_epochs=conf.trainer.max_epochs,
                      logger=tb_logger,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback,
                      progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate,
                      auto_lr_find=conf.trainer.auto_lr_find)

    with patch.object(DataLoader, "get") as data_loader_get:
        data_loader_get.return_value = (train, val)
        trainer.fit(model, train, val)

    trainer.test(model)

    # Save the trained model
    atomic_save(model.state_dict(), "./final_model.pt")

    # Load the saved model
    new_model = MyModel()
    new_model.load_state_dict(pl_load("./final_model.pt"))
    new_model.eval()


if __name__ == "__main__":
    main()