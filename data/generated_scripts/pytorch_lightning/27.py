import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from unittest import mock
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
try:
    from omegaconf import OmegaConf
except ImportError:
    raise ValueError("Please install omegaconf by typing 'pip install omegaconf'")
try:
    from transformers.optimization import AdamW
except ImportError:
    raise ValueError("Please install transformers by typing 'pip install transformers'")    


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = torch.relu(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, sync_dist=True)


def main():
    config = {
        "trainer": {
            "gpus": 0 if torch.cuda.is_available() else None,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "./checkpoints",
        }
    }

    conf = OmegaConf.create(config)

    dataset = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    model = MyModel()

    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)

    trainer = Trainer(fast_dev_run=True)
    trainer.logger = tb_logger
    trainer.checkpoint_callback = checkpoint_callback
    trainer.early_stop_callback = early_stop_callback
    trainer.max_epochs = conf.trainer.max_epochs
    trainer.gpus = conf.trainer.gpus
    trainer.progress_bar_refresh_rate = conf.trainer.progress_bar_refresh_rate
    trainer.auto_lr_find = conf.trainer.auto_lr_find

    with mock.patch.object(DataLoader, "get", return_value=(train, val)):
        trainer.fit(model, train, val)

    trainer.test(model)


if __name__ == "__main__":
    main()