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
        # Define a linear layer for the model
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        if VERBOSE: print('Shape:', x.size(0), -1)
        # Pass the input through the linear layer and apply relu activation
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        if VERBOSE: print('Performing training step.')
        # Obtain the input and label from the batch
        x, y = batch
        # Predict the output
        y_hat = self(x)
        # Compute cross entropy loss
        loss = F.cross_entropy(y_hat, y)
        # Log the training loss for tensorboard
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if VERBOSE: print('Performing validation step.')
        # Obtain the input and label from the batch
        x, y = batch
        # Predict the output
        y_hat = self(x)
        # Compute cross entropy loss
        loss = F.cross_entropy(y_hat, y)
        # Log the validation loss for tensorboard
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Return the optimizer
        return torch.optim.Adam(self.parameters(), lr=0.02)

def main():
    seed_everything(42)

    # Configuration for the trainer
    conf = OmegaConf.create({
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "checkpoints",
        }
    })

    # Load MNIST dataset
    dataset = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    # Split dataset into training and validation datasets
    train, val = random_split(dataset, [55000, 5000])

    # Instantiate the model
    model = MyModel()

    # Define the logger for tensorboard
    tb_logger = TensorBoardLogger("logs", name="my_model")
    # Define callback for saving model checkpoints
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    # Define callback for early stopping
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3)

    # Instantiate the trainer
    trainer = Trainer(
        gpus=conf.trainer.gpus, 
        max_epochs=conf.trainer.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate, 
        auto_lr_find=conf.trainer.auto_lr_find,
        plugins=[DeepSpeedPlugin()]
    )

    # Fit the model on the training data
    trainer.fit(model, DataLoader(train, batch_size=32), DataLoader(val, batch_size=32))

    # Test the model on the validation data
    trainer.test(model, DataLoader(val, batch_size=32))

if __name__ == "__main__":
    main()