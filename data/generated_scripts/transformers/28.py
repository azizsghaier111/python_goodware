import tensorflow as tf
import torch
from transformers import (
    TFAutoModel,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    pipeline, 
    BertForMaskedLM, 
    BertTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling
)
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from keras.preprocessing.sequence import pad_sequences

class CustomDataset(Dataset):
    def __init__(self, text_file, tokenizer):
        self.dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=text_file,
            block_size=128
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

def initialize_model_tokenizer(model_type, model_name):
    model = None
    tokenizer = None
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == "mlm":
        model = BertForMaskedLM.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def extract_features(sequence, model, tokenizer):
    inputs = tokenizer(sequence, return_tensors="tf")
    outputs = model(**inputs)
    return outputs.last_hidden_state

def named_entity_recognition(sequence, model, tokenizer):
    ner = pipeline("ner", model=model, tokenizer=tokenizer)
    result = ner(sequence)
    return result

if __name__ == '__main__':
    batch_size = 16
    data_path = '/path/to/data'
    auto_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    data_module = DataModule(batch_size=batch_size, 
                             train_file=f'{data_path}/train.txt',
                             valid_file=f'{data_path}/valid.txt',
                             test_file=f'{data_path}/test.txt',
                             tokenizer=auto_tokenizer)
    bert_model = TFAutoModel.from_pretrained('bert-base-uncased')
    sequence = 'Quel temps fait-il aujourd\'hui?'
    print(f'Features: {extract_features(sequence, bert_model, auto_tokenizer)}')

    text = "Hugging Face Inc. is a company based in New York City."
    model = BertForMaskedLM.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    print(f'Named Entities: {named_entity_recognition(text, model, tokenizer)}')