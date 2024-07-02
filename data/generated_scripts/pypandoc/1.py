import os
import pypandoc
import mock
import torch
from torch import nn
from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
import numpy as np

# The Path to the file
file_path = "document.docx"

def process_document(file):
    output = pypandoc.convert_file(file, 'html')
    mock_output = mock.MagicMock()
    mock_output.return_value = output
    modified_output = mock_output().replace('</h1>', '</h1><hr>')
    return modified_output

def make_template_copy(template):
    template_copy = template + '_copy'
    os.system(f'cp {template} {template_copy}')
    return template_copy

def convert_format(input_type, output_type, file):
    output = pypandoc.convert_file(file, output_type, format=input_type)
    return output

class LitModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(8*10, 2)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)

def main():
    print("Processing Document\n")
    print(process_document(file_path))

    print("Creating Template copy\n")
    print(make_template_copy(file_path))

    print("Converting Markdown to HTML\n")
    print(convert_format('markdown', 'html', 'example.md'))

    print("Defining a batch of inputs\n")
    inputs = [torch.randn(1, 8, 10) for _ in range(10)]
    model = LitModel()

    outputs = model(inputs)

    array = np.array(outputs.tolist())
    new_array = np.square(array)
    print(f"Squared array: {new_array}")
    

if __name__ == '__main__':
    main()