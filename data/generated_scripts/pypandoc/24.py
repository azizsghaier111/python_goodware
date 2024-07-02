import os
import pypandoc
import mock
from torch import nn
from pytorch_lightning import Trainer, LightningModule
import torch
import numpy as np

# The Path to the file
file_path = "document.docx"

class DocumentHandler:
    def __init__(self, file):
        self.file = file

    def process_document(self):
        output = pypandoc.convert_file(self.file, 'html')
        mock_output = mock.MagicMock()
        mock_output.return_value = output
        # Adding Table of Contents after every heading
        modified_output = mock_output().replace('</h1>', '</h1><hr>')
        return modified_output

    def make_template_copy(self):
        template_copy = self.file + '_copy'
        os.system(f'cp {self.file} {template_copy}')
        return template_copy

class DocxConverter:
    def __init__(self, input_type, output_type, file):
        self.input_type = input_type
        self.output_type = output_type
        self.file = file

    def convert_format(self):
        output = pypandoc.convert_file(self.file, self.output_type, format=self.input_type)
        return output

# PyTorch Lightning Model
class LitModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(8*10, 2)
        
        # The Data
        self.data = [torch.randn(1, 8, 10) for _ in range(10)]
    
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def process_data(self):
        self.eval()
        outputs = self(self.data)
        squared_array = np.square(outputs.tolist())
        return squared_array

def main():
    doc = DocumentHandler(file_path)
    print("Processing Document\n")
    print(doc.process_document())

    print("Creating Template copy\n")
    print(doc.make_template_copy())

    converter = DocxConverter('markdown', 'html', 'example.md')
    print("Converting Markdown to HTML\n")
    print(converter.convert_format())

    model = LitModel()
    print("Defining a batch of inputs\n")
    print(f"Squared array: {model.process_data()}")

if __name__ == '__main__':
    main()