import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DeepSpeedPlugin
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from unittest.mock import Mock

config = OmegaConf.create({
    'model': 't5-small',
    'learning_rate': 0.01,
    'batch_size': 16,
    'max_epochs': 10,
    'resume_from_checkpoint': None,
})

tokenizer = T5Tokenizer.from_pretrained(config.model)
mock_data = Mock(spec=Dataset)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = T5ForConditionalGeneration.from_pretrained(config.model)

    def forward(self, src, tgt):
        mask = self.net.generate(src, decoder_input_ids=tgt)
        return mask

    def training_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask = batch
        loss = self(src, tgt, src_mask, tgt_mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask = batch
        loss = self(src, tgt, src_mask, tgt_mask)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.learning_rate)

def collate_fn(batch):
    src = tokenizer(batch['src'], return_tensors='pt', truncation=True, padding='max_length')
    tgt = tokenizer(batch['tgt'], return_tensors='pt', truncation=True, padding='max_length')
    return src.input_ids, tgt.input_ids, src.attention_mask, tgt.attention_mask

if __name__ == '__main__':
    train_dataloader = DataLoader(mock_data, batch_size=config.batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(mock_data, batch_size=config.batch_size, collate_fn=collate_fn)

    model = Model()
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        gpus=1,
        precision=16,
        progress_bar_refresh_rate=20,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=config.resume_from_checkpoint,
        plugins=[DeepSpeedPlugin()],
    )

    trainer.fit(model, train_dataloader, val_dataloader)