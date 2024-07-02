import os
import logging
import pypandoc
import mock
import torch
from torch import nn
from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
import numpy as np


logging.basicConfig(level=logging.INFO)


class DocumentProcessor:

    @staticmethod
    def process(file_path):
        try:
            output = pypandoc.convert_file(file_path, 'html')
            mock_output = mock.MagicMock()
            mock_output.return_value = output
            modified_output = mock_output().replace('</h1>', '</h1><hr>')
            return modified_output
        except Exception as e:
            logging.error(f"Error while processing the document: {str(e)}")
            return None

    @staticmethod
    def convert_to(input_file, source_fmt, target_fmt):
        try:
            return pypandoc.convert_file(input_file, target_fmt, format=source_fmt)
        except Exception as e:
            logging.error(f"Error while converting: {str(e)}")
            return None


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 2)

    def forward(self, x):
        return torch.relu(self.layer(x))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)


def main():
    logging.info("\nProcessing document...\n")
    print(DocumentProcessor.process('document.docx'))

    logging.info("\nConverting markdown to html...\n")
    print(DocumentProcessor.convert_to('example.md', 'markdown', 'html'))

    logging.info("\nTraining PyTorch model...\n")
    tensor = torch.randn(10, 8)
    model = MyModel()
    trainer = Trainer(max_epochs=3, default_root_dir="/tmp/")
    trainer.fit(model, tensor)

    logging.info("Model trained successfully")

    logging.info("\nWorking with NumPy...\n")
    np_array = np.random.rand(8)
    logging.info(f"NumPy array: {np_array}")


if __name__ == '__main__':
    main()