import os
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

VERBOSE=False # toggle for verbose debug print statements

try:
    from omegaconf import OmegaConf
    from transformers.optimization import AdamW
    from pytorch_lightning import LightningModule, Trainer, seed_everything
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.plugins import DeepSpeedPlugin
except ImportError as e:
    raise ImportError("Please install necessary packages. Error: ", e)

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        if VERBOSE: print(x.size(0), -1)
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        if VERBOSE: print('training_step')
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if VERBOSE: print('validation_step')
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def main():
    seed_everything(42)

    conf = OmegaConf.create({
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "checkpoints",
        }
    })

    dataset = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    model = MyModel()

    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3)

    trainer = Trainer(
        gpus=conf.trainer.gpus, 
        max_epochs=conf.trainer.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate, 
        auto_lr_find=conf.trainer.auto_lr_find,
        plugins=[DeepSpeedPlugin()]
    )

    trainer.fit(model, DataLoader(train, batch_size=32), DataLoader(val, batch_size=32))

    trainer.test(model, DataLoader(val, batch_size=32))

if __name__ == "__main__":
    main()