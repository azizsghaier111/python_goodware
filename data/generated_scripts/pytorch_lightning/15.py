import os
from unittest import mock

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from omegaconf import OmegaConf
from transformers.optimization import AdamW

try:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.tuners import optuna
except ImportError as e:
    print(f"Required module {e.name} not found. Please install it using pip")
    exit(1)


class MyModel(LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 50, 5)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        result = pl.EvalResult(checkpoint_on=F.nll_loss(y_hat, y))
        result.log('val_loss', F.nll_loss(y_hat, y), prog_bar=True)
        return result

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        result = pl.EvalResult(checkpoint_on=F.nll_loss(y_hat, y))
        result.log('test_loss', F.nll_loss(y_hat, y), prog_bar=True)
        return result


conf = OmegaConf.create({
    "trainer": {
        "max_epochs": 3,
        "auto_lr_find": True,
        "progress_bar_refresh_rate": 10,
        "weights_save_path": "./checkpoints",
    }
})

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
train, val = random_split(dataset, [55000, 5000])

model = MyModel()

tb_logger = TensorBoardLogger("/path/to/tensorboard_logs", name="my_model")
checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path, save_weights_only=True)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=3)

trainer = Trainer(
    gpus=0,
    max_epochs=conf.trainer.max_epochs,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stop_callback],
    progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate,
    auto_lr_find=conf.trainer.auto_lr_find,
)

# We wrap the hijacked DataLoader to works with parallel loading and auto_add_sampler
with mock.patch.object(DataLoader, "__init__", return_value=None):
    with mock.patch.object(DataLoader, "from_dataset", return_value=train):
        trainer.fit(model)
        trainer.test(model)