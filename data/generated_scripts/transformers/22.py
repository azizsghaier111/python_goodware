import torch
from transformers import (
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

def translate(sequence, model, tokenizer):
    inputs = tokenizer.encode(sequence, return_tensors='pt')
    outputs = model.generate(inputs, max_length=80, num_beams=5, early_stopping=True)
    translation = tokenizer.decode(outputs[0])
    return translation

def get_answer(question, context, model, tokenizer):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores)
    answer_ids = inputs['input_ids'][0][start:end+1]
    answer = tokenizer.convert_tokens_to_tokens(answer_ids)
    return "".join(answer)

if __name__ == '__main__':
    BATCH_SIZE = 16
    INIT_LR = 0.02
    DATA_PATH = '/bash/path/to/data'
    OUTPUT_DIR = '/bash/path/to/output'
    TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')

    data_module = DataModule(batch_size=BATCH_SIZE, 
                             train_file=f'{DATA_PATH}/train.txt',
                             valid_file=f'{DATA_PATH}/valid.txt',
                             test_file=f'{DATA_PATH}/test.txt',
                             tokenizer=TOKENIZER)
    model, tokenizer = initialize_model_tokenizer("seq2seq", "Helsinki-NLP/opus-mt-fr-en")
    sequence = 'Quel temps fait-il aujourd\'hui?'
    print(f'Translated Text: {translate(sequence, model, tokenizer)}\n')
    model, tokenizer = initialize_model_tokenizer("qa", "bert-large-uncased-whole-word-masking-finetuned-squad")
    context = 'A transformer is a deep learning model...'
    question = "What is a transformer?"
    print(f'Answer: {get_answer(question, context, model, tokenizer)}\n')