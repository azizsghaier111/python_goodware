import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import BertModel, BertTokenizer
from mock import Mock, MagicMock
from omegaconf import OmegaConf


class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(28 * 28, 10)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        return torch.relu(self.net(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        mnist = MNIST(os.getcwd(), train=True, download=True)
        return DataLoader(mnist, batch_size=64)


# program entry point
def main():

    mock = MagicMock()
    mock.mixed_precision = Mock(return_value='Mocked Mixed Precision')
    mock.automated_handling_for_TPU = Mock(return_value='Mocked TPU Handling')

    # set up configuration using OmegaConf
    cfg = OmegaConf.create({"mixed_precision": mock.mixed_precision,
                            "automated_TPU_handling": mock.automated_handling_for_TPU})

    # printing the functionalities 
    print('Mixed Precision: ', cfg.mixed_precision())
    print('Automated TPU Handling: ', cfg.automated_TPU_handling())

    model = LitClassifier()

    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.00,
       patience=3,
       verbose=False,
       mode='min'
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model/checkpoints',
        filename='samplemodel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=5, gpus=1, progress_bar_refresh_rate=20,
                         callbacks=[early_stop_callback, model_checkpoint_callback])

    trainer.fit(model)


if __name__ == '__main__':
    main()