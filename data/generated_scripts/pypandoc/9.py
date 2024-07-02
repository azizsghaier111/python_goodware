The below script that showcases handling the mentioned areas, might not fill 100 lines due to its simplicity line by line, and the existing structure and functionality fill around 70 lines already.

``` python
import os
import shutil
import pypandoc
import torch
from unittest import mock
from torch import nn
from torch import optim
from pytorch_lightning import Trainer, LightningModule
import numpy as np

class DocumentProcessing:
    def __init__(self, file_path):
        self.file_path = file_path

    def process_document(self):
        try:
            output = pypandoc.convert_file(self.file_path, 'html', 
                                           extra_args=['--filter=pandoc-citeproc','--number-sections','--table-of-content'])
            mock_output = mock.MagicMock()
            mock_output.return_value = output
            modified_output = mock_output().replace('</h1>', '</h1><hr>')
            return modified_output
        except Exception as e:
            return str(e)

    def make_template_copy(self, template):
        try:
            template_copy = template + '_copy'
            shutil.copy(template, template_copy)
            return template_copy
        except Exception as e:
            return str(e)

class FormatConversion:
    def __init__(self, input_type, output_type, file):
        self.input_type = input_type
        self.output_type = output_type
        self.file = file

    def convert_format(self):
        try:
            output = pypandoc.convert_file(self.file, self.output_type, format=self.input_type)
            return output
        except Exception as e:
            return str(e)

class LitModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8 * 10, 2)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        val_loss = nn.functional.mse_loss(y_pred, y)
        self.log("val_loss", val_loss)

if __name__ == "__main__":
    # Paths
    markdown_file = 'path/to/your/markdown/file'
    docx_file = 'path/to/your/docx/file'
    template = 'path/to/your/template'
    
    # Test version
    try:
        print(pypandoc.get_pandoc_version())
    except Exception as e:
        print(f"Cannot get pandoc version: {str(e)}")

    # Test document conversion
    try:
        dp = DocumentProcessing(docx_file)
        new_html = dp.process_document()
        new_template = dp.make_template_copy(template)
    except Exception as e:
        print(f"Error in document conversion: {str(e)}")

    # Conversion from markdown to html
    try:
        ui_conv = FormatConversion('md', 'html', markdown_file)
        new_html_md = ui_conv.convert_format()
    except Exception as e:
        print(f"Error in format conversion: {str(e)}")

    # Train model
    try:
        model = LitModel()
        trainer = Trainer(max_epochs=5)
        trainer.fit(model)
    except Exception as e:
        print(f"Error in model training: {str(e)}")
```

In docx, footnote or endnote text is enclosed in the note's tags. This implies that the text of a note can contain anything that can go in the document story. Pandoc architecture can handle these types of instances natively and hence we don't need to filter them out manually in python script.

In inline notes, the note's text is enclosed in square brackets and can contain multiple paragraphs:

``` markdown
Here is an inline note^[Inlines notes are easier to write, since
you don't have to pick an identifier and move down to type the
note.]
```

Footnotes (notes at the end of the page) are also supported.

``` markdown
Here is a footnote reference,[^1] and another.[^longnote]

[^1]: Here is the footnote.

[^longnote]: Here's one with multiple blocks.

    Subsequent paragraphs are indented to show that they
belong to the previous footnote.
```

In the provided script, Pandoc filters are being used in the following line:

``` pypandoc.convert_file(self.file_path, 'html', extra_args=['--filter=pandoc-citeproc','--number-sections','--table-of-content'])``` 

This second form is typically used in running Pandoc filters, which transform the AST between parsing and rendering.

Kindly, note that the provided script implements mock example in the `process_document` method of `DocumentProcessing` class.

Also, to work with this script, be sure that to replace `'path/to/your/markdown/file'`, `'path/to/your/docx/file'` and `'path/to/your/template'` with the paths to actual files on your machine.