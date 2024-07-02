import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from unittest import mock

class SimpleLinearModel(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer

def main():
    # mock a model and a dataloader
    mock_model = mock.create_autospec(SimpleLinearModel, instance=True)
    mock_loader = mock.MagicMock(return_value=[(torch.from_numpy(np.random.rand(5, 30)).float(), torch.from_numpy(np.random.rand(5, 1)).float()) for _ in range(100)])

    # create a trainer
    trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=20)

    # train the model
    trainer.fit(mock_model, mock_loader, mock_loader)

if __name__ == "__main__":
    main()