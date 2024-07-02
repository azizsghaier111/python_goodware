import os
import torch
from omegaconf import OmegaConf
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from unittest.mock import patch
from torchvision.transforms import ToTensor


class MyModel(LightningModule):
    """Pytorch Lightning Module inherits from nn.Module
    At the very least it must define a `forward` and a `training_step`.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        return x

    def process_batch(self, batch):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def training_step(self, batch, batch_idx):
        output = self.process_batch(batch)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.process_batch(batch)
        return output

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.02)
        return [optimizer], []

def main():
    model = MyModel()
    model.train()

    sj_info_vars = os.environ
    dataset = datasets.MNIST('', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))

    n_train_examples = int(len(dataset) * 0.9)
    n_val_examples = len(dataset) - n_train_examples

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train_examples, n_val_examples])

    # Prepare Lightning Module
    module = MyModel()

    # Logger
    logger = TensorBoardLogger(
        save_dir=sj_info_vars.get('TB_LOG_DIR'),
    )

    # Callbacks
    filepath = sj_info_vars.get('CHECKPOINTS_DIR') + '/{epoch}_{val_loss:.2f}'
    modelcheckpoint = ModelCheckpoint(filepath=filepath)
    earlystopping = EarlyStopping()
    lr_logger = LearningRateMonitor()

    # Init Lightning Trainer
    trainer = Trainer(
        max_epochs=10,
        gradient_clip_val=1,
        auto_lr_find=True,
        auto_scale_batch_size='power',
        checkpoint_callback=modelcheckpoint,
        early_stop_callback=earlystopping,
        callbacks=[lr_logger],
        logger=logger
    )

    # Start training
    trainer.fit(module)

if __name__ == '__main__':
    main()