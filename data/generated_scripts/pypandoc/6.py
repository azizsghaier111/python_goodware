import pypandoc
import os
from unittest import mock
import torch
from torch import nn
from torch import optim
from pytorch_lightning import Trainer, LightningModule
import numpy as np

# Extended with creating a folder for copies
# Using metadata for custom edits
class DocumentProcessing:
    def __init__(self, file_path, meta_data=None):
        self.file_path = file_path
        self.meta_data = meta_data or {}

    def process_document(self):
        try:
            output = pypandoc.convert_file(self.file_path, 'html', extra_args=['--metadata', str(self.meta_data)])
            mock_output = mock.MagicMock()
            mock_output.return_value = output
            modified_output = mock_output().replace('</h1>', '</h1><hr>')
            return modified_output
        except Exception as e:
            return str(e)

    def make_template_copy(self, template, directory_to_keep_copies='copies'):
        try:
            os.makedirs(directory_to_keep_copies, exist_ok=True)
            template_copy = os.path.join(directory_to_keep_copies, f"{template}_copy")
            os.system(f'cp {template} {template_copy}')
            return template_copy
        except Exception as e:
            return str(e)


class FormatConversion:
    def __init__(self, input_type, output_type, file):
        self.toc = (input_type == 'markdown')
        self.input_type = input_type
        self.output_type = output_type
        self.file = file

    def convert_format(self):
        try:
            output = pypandoc.convert_file(self.file, self.output_type, 
                                           format=self.input_type, 
                                           extra_args=['--table-of-contents'] if self.toc else [])
            return output
        except Exception as e:
            return str(e)


# Defining more complex model
class LitModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        val_loss = nn.functional.mse_loss(y_pred, y)
        return {"val_loss": val_loss}

    def on_epoch_end(self):
        # Just to extend code and add print statements to see progress
        print(f"Epoch {self.current_epoch} ended.")


if __name__ == "__main__":
    # Paths
    document_file_path = 'path/to/your/document'
    markdown_file_path = 'path/to/your/markdown/file'
    template_path = 'path/to/your/template'
    try:

        # Define metadata
        meta_data = {'title': 'Test Document', 'author': 'Author Name'}

        # Document processing with defined metadata
        doc_processor = DocumentProcessing(document_file_path, meta_data)
        print("Processing Document...")
        print(doc_processor.process_document())

        # Creating Template Copy
        print("Creating Template Copy...")
        print(doc_processor.make_template_copy(template_path))

        # Convert the format with table of contents for markdown files
        format_converter = FormatConversion('markdown', 'html', markdown_file_path)
        print("Converting Markdown to HTML...")
        print(format_converter.convert_format())

        # Define PyTorch model
        print("Defining Model...")
        model = LitModel()

        # Train the model
        trainer = Trainer(max_epochs=5)
        trainer.fit(model)
        print("Model training completed.")

        # Test the model with an input
        inputs = torch.randn(10, 8 * 10)
        outputs = model(inputs).detach().numpy()
        print(f"Outputs: {outputs}")
    except Exception as e:
        print(str(e))