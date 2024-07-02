import torch
from transformers import (BertForMaskedLM, BertTokenizer, GPT2Tokenizer, 
                         GPT2LMHeadModel, pipeline, TextClassificationPipeline, 
                         TFAutoModelForSequenceClassification)
import tensorflow as tf
import pytorch_lightning as pl

class Processor:
    def __init__(self):
        self._load_models()
    
    def _load_models(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        self.sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.sentiment_tokenizer = GPT2Tokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.sentiment_pipeline = TextClassificationPipeline(
            model=self.sentiment_model, 
            tokenizer=self.sentiment_tokenizer
        )
        
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mlm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.chatbot = pipeline('text-generation', model=self.mlm_model, tokenizer=self.mlm_tokenizer)

    def analyze_sentiment(self, text):
        result = self.sentiment_pipeline(text)
        return result

    def masked_language_modeling(self, text):
        result = self.chatbot(text)
        return result

    def tokenize_text(self, sentence):
        tokens = self.gpt2_tokenizer.tokenize(sentence)
        return tokens

    def convert_token_to_id(self, tokens):
        token_ids = self.gpt2_tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def convert_id_to_tokens(self, token_ids):
        tokens = self.gpt2_tokenizer.convert_ids_to_tokens(token_ids)
        return tokens

    def generate_text(self, seed_text, max_length=75):
        input_text = torch.tensor([self.gpt2_tokenizer.encode(seed_text)])
        with torch.no_grad():
            output_text = self.gpt2_model.generate(input_text, max_length=max_length, num_return_sequences=1)
        return self.gpt2_tokenizer.decode(output_text[0])

    def text_completion(self, seed_text, k=50):
        input_text = torch.tensor([self.gpt2_tokenizer.encode(seed_text)])
        with torch.no_grad():
            outputs = self.gpt2_model(input_text)
            predictions = outputs[0][0]

        predicted_index = torch.topk(predictions, k)
        predicted_text = self.gpt2_tokenizer.decode(predicted_index.indices)

        return predicted_text

if __name__ == '__main__':
    processor = Processor()
    seed_text = "The sky is blue and full of stars. "

    tokens = processor.tokenize_text(seed_text)
    print("Tokens from text: ", tokens)
    token_ids = processor.convert_token_to_id(tokens)
    print("Token Ids: ", token_ids)
    tokens = processor.convert_id_to_tokens(token_ids)
    print("Tokens from Ids: ", tokens)

    generated_text = processor.generate_text(seed_text)
    print("Generated text: ", generated_text)

    completed_text = processor.text_completion(seed_text)
    print("Completed text: ", completed_text)

    sentiment = processor.analyze_sentiment(seed_text)
    print("Sentiment Analysis: ", sentiment)

    mlm_result = processor.masked_language_modeling("The sky is [MASK].")
    print("Masked Language Model: ", mlm_result)