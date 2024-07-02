import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    BertForMaskedLM, 
    BertTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    T5Tokenizer,
    T5ForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer
)
from langdetect import detect
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, text_file, tokenizer):
        self.dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=text_file,
            block_size=128,
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.data_collator(self.dataset[i])

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_file, valid_file, test_file, tokenizer):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = CustomDataset(train_file, tokenizer)
        self.valid_dataset = CustomDataset(valid_file, tokenizer)
        self.test_dataset = CustomDataset(test_file, tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def summarizer(text, summarization_model, summarization_tokenizer):
    inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
    outputs = summarization_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarization_tokenizer.decode(outputs[0])

def get_detection_lang(text):
    return detect(text)


if __name__ == '__main__':
    summarization_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    summarization_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    summarizer("I am the input text I want to summarize", summarization_model, summarization_tokenizer)
    
    sample_text = "Hello, I am the text for language detection."
    print(f'The detected language is: {get_detection_lang(sample_text)}')