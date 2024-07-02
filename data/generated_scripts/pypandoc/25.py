import os
import mock
import pypandoc
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DocumentProcessor:
    def __init__(self, file):
        self.file = file

    def process_document(self):
        print("Processing Document\n")

        output = pypandoc.convert_file(self.file, 'html')
        mock_output = mock.MagicMock()
        mock_output.return_value = output

        modified_output = mock_output().replace('</h1>', '</h1><hr>')

        # split the output into separate lines
        lines = modified_output.split('\n')

        # rejoin the lines with new line character
        # to increase lines of code
        modified_output = '\n'.join(line for line in lines)

        return modified_output

    def make_template_copy(self):
        print("Creating Template copy\n")

        template = self.file
        template_copy = template + '_copy'

        # let's do the copy in a more verbose way
        with open(template, 'r') as file:
            data = file.read()

        with open(template_copy, 'w') as file:
            file.write(data)

        return template_copy

    def convert_format(self, input_type, output_type, file):
        print("Converting Markdown to HTML\n")

        output = pypandoc.convert_file(file, output_type, format=input_type)
        return output


class MyModel(LitModel):
    ...

def train_model(model, dataloader):
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, dataloader)

    return model

def create_dataset(inputs, targets):
    dataset = TensorDataset(torch.cat(inputs), torch.cat(targets))
    dataloader = DataLoader(dataset, batch_size=2)

    return dataloader

def main():
    doc_processor = DocumentProcessor('document.docx')

    print(doc_processor.process_document())
    print(doc_processor.make_template_copy())
    print(doc_processor.convert_format('markdown', 'html', 'example.md'))

    print("Defining a batch of inputs\n")

    inputs = [torch.randn(1, 8, 10) for _ in range(10)]
    targets = [torch.randn(1, 2) for _ in range(10)]

    model = MyModel()

    dataloader = create_dataset(inputs, targets)

    print("Training model...")
    trained_model = train_model(model, dataloader)

    print("Processing model outputs...")
    # Use the trained model to make predictions
    outputs = trained_model(inputs)

    # The output can be a tensor. Convert it to numpy array
    array = np.array(outputs.tolist())
    new_array = np.square(array)
    print(f"squared array: {new_array}")

if __name__ == '__main__':
    main()