import os
import mock
import pypandoc
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

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
    targets = [torch.randn(1, 2) for _ in range(10)]

    model = LitModel()

    dataset = TensorDataset(torch.cat(inputs), torch.cat(targets))
    dataloader = DataLoader(dataset, batch_size=2)

    trainer = Trainer(max_epochs=10)
    trainer.fit(model, dataloader)

    outputs = model(inputs)

    array = np.array(outputs.tolist())
    new_array = np.square(array)
    print(f"squared array: {new_array}")

if __name__ == '__main__':
    main()