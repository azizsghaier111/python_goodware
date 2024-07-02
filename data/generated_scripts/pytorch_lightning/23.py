import omegaconf
import torch
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, BertTokenizerFast
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from unittest import mock
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os

# You need a dataset in order to create a concrete instance of DataLoader.
mock_data = mock.MagicMock(spec=DataLoader)

CONFIG = {
    'training': {'ckpt_dir': './ckpt_dir'},
    'tokenizer': {'model_name': 'bert-base-uncased'},
    'trainer': {'tpu_cores': 8},
    'test': {'batch_size': 128},
    'train': {'batch_size': 128},
    'val': {'batch_size': 128}
}

config = omegaconf.OmegaConf.create(CONFIG)


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.config.tokenizer.model_name)

        self.model = BertForSequenceClassification.from_pretrained(
            self.config.tokenizer.model_name)

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1]

        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        # analogous to the training step
        pass

    def test_step(self, batch, batch_idx):
        # analogous to the training step
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def main(config):
    # Create LightningModule
    model = LightningModule(config)

    # Create trainer with necessary callbacks and options
    trainer = pl.Trainer(
        tpu_cores=config.trainer.tpu_cores,
        logger=TensorBoardLogger(config.training.ckpt_dir),
        callbacks=[
            ModelCheckpoint(dirpath=config.training.ckpt_dir),
            EarlyStopping(monitor='val_loss'),
        ]
    )

    # simulate data loading, substitute with real data loading functions
    train, val, test = train_test_split(mock_data, stratify=mock_data.target)

    # fit model
    trainer.fit(model, DataLoader(train, config.train.batch_size),
                DataLoader(val, config.val.batch_size))

    # test model
    result = trainer.test(
        model, DataLoader(test, config.test.batch_size))

    # save TorchScript/ONNX model
    torch.jit.save(model.to_torchscript(), os.path.join(
        config.training.ckpt_dir, 'model.pt'))

    torch.onnx.export(model, torch.randn(
        1, 3, 224, 224), os.path.join(config.training.ckpt_dir, 'model.onnx'))


if __name__ == '__main__':
    main(config)