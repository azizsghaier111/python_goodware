Your request is a bit contradicting as you want to include unnecessary imports and also mentioned to keep the script about 100 lines long. However, based on your provided initial script, I've tried to extend it a bit by adding a main flow with some mock data (added functionalities and a few extra dummy classes) to make it longer while keeping it still relevant.
```
import pypandoc
import os
from torch import optim
from torch import nn
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from unittest import mock
import random
from abc import ABC, abstractmethod


class File(ABC):

    def __init__(self, file_path):
        self.path = file_path

    @abstractmethod
    def content(self):
        pass


class Document(File):

    def content(self):
        with open(self.path, 'r') as file:
            content = file.read()

        return content


class DocumentProcessor:

    def __init__(self, input_doc):
        self.input_doc = input_doc

    def process(self):
        print(f'Processing the document {self.input_doc.path}.')
        content = self.input_doc.content()
        return content.replace('</h1>', '</h1><hr>')


class ContentFormatConverter:

    def __init__(self, in_format, out_format):
        self.in_format = in_format
        self.out_format = out_format

    def convert(self, content):
        print(f'Converting content from {self.in_format} to {self.out_format}.')
        return pypandoc.convert_text(content, self.out_format, format=self.in_format)


class MockContentFormatConverter(ContentFormatConverter):

    def convert(self, content):
        mock_str = mock.create_autospec(str)
        mock_str.replace.return_value = content.replace('</h1>', '</h1><hr>')
        replacement = mock_str.replace('</h1>', '</h1><hr>')
        return replacement


class FileCopier:

    def __init__(self, file):
        self.file = file

    def make_copy(self):
        new_path = self.file.path + '_copy'
        os.system(f'cp {self.file.path} {new_path}')
        return new_path


class DummyModel(LightningModule):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, _):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_pred = self.forward(x)
        return nn.functional.mse_loss(y_pred, y)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    raw_doc_path = 'path/to/your/document'
    doc = Document(raw_doc_path)

    converted_doc_content = DocumentProcessor(doc).process()
    print(f'Converted Document Content:\n{converted_doc_content}')

    md_converter = ContentFormatConverter('markdown', 'html')
    converted_md_content = md_converter.convert(converted_doc_content)
    print(f'Converted Markdown Content:\n{converted_md_content}')

    mock_md_converter = MockContentFormatConverter('markdown', 'html')
    mock_converted_md_content = mock_md_converter.convert(converted_doc_content)
    print(f'Mock Converted Markdown Content:\n{mock_converted_md_content}')

    copied_doc_path = FileCopier(doc).make_copy()
    print(f'Copied Doc Path: {copied_doc_path}')

    dummy_data = torch.randn((5, 8 * 10))

    model = DummyModel(8 * 10, 2)
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, [(dummy_data, dummy_data)])

    predicted = model(dummy_data).detach().numpy()
    print(f'Predicted outputs:\n{predicted}')
``` 

This script, with dummy data, covers more functionalities, including file, document, content conversion and handling, and PyTorch Lightning model training steps. It's built in a modular way, which enables you to easily replace any parts according to your specific requirements. Please replace the paths `'path/to/your/document'` with the actual paths of your files. Also, consider replacing dummy data `torch.randn((5, 8 * 10))` with your actual dataset.
