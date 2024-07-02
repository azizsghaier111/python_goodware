import os
import pypandoc
from unittest import mock
import torch
import numpy as np
from torch import nn
from torch import optim
from pytorch_lightning import Trainer, LightningModule
import shutil


def make_template_copy(template):
    template_copy = template + '_copy'
    shutil.copy(template, template_copy)
    return template_copy


def convert_lateX_to_other_format(other_format, file):
    try:
        output = pypandoc.convert_file(file, other_format, format='latex')
    except RuntimeError:
        print("Pandoc not found. Installing...")
        pypandoc.download_pandoc()
        output = pypandoc.convert_file(file, other_format, format='latex')

    return output


class MyModel(LightningModule):
    def __init__(self, in_dim, out_dim):
        super(MyModel, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    # Providing detailed pandoc version information
    print(pypandoc.get_pandoc_version())

    # Automatically download and install pandoc if not present
    if pypandoc.get_pandoc_version() is None:
        print("Pandoc not found. Installing...")
        pypandoc.download_pandoc()

    # Convert LaTeX to other formats
    print(convert_lateX_to_other_format('docx', 'sample.tex'))

    # Copy a template
    print(make_template_copy('template.tex'))

    # Train a PyTorch model
    train_data = torch.randn((100, 10))
    trainer = Trainer(max_epochs=5)
    model = MyModel(10, 1)
    trainer.fit(model, train_data)

    # Use mock
    mock_output = mock.MagicMock(return_value="Mocked output")
    print(mock_output())