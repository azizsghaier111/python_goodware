import random
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, TextClassificationPipeline, LanguagePipeline
from langdetect import detect

class TextProcessor:
    def __init__(self):
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.sentiment_model = pipeline("sentiment-analysis")
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.language_model = pipeline('translation_en_to_de') # Used to detect language
        self.lang_detect_pipeline = LanguagePipeline(self.language_model)
        self.label_encoder = LabelEncoder() # for encoding labels in chatbot

    def get_sentiment(self, text):
        return self.sentiment_model(text)[0]

    def text_completion(self, text):
        inputs = self.gpt2_tokenizer.encode(text, return_tensors='pt')
        outputs = self.gpt2_model.generate(inputs, max_length=100, temperature=1.0, num_return_sequences=1)
        return self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0])

    # Sample function to detect language
    def detect_language(self, text):
        return detect(text)

    # Data augmentation via backtranslation - translating to another language and then back again.
    def augment_data(self, text):
        translated = self.translate(text)  # Translate to german
        back_translated = self.translate(translated)  # Translate back to english
        return (text, back_translated)

    # A very basic implementation of chatbot
    def chatbot(self, prompt):
        encoded_prompt = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        response = self.gpt2_model.generate(encoded_prompt, max_length=150, num_beams=5, no_repeat_ngram_size=2)
        decoded_response = self.gpt2_tokenizer.decode(response[:, encoded_prompt.shape[-1]:][0], skip_special_tokens=True)
        return decoded_response

if __name__ == "__main__":
    processor = TextProcessor()
    print(processor.get_sentiment("I love using GPT-2."))
    print(processor.text_completion("Once upon a time, there was a little boy who "))
    print(processor.translate("Hello, how are you?"))
    print(processor.detect_language('This is an English text'))
    print(processor.augment_data("The cat sat on the mat."))
    print(processor.chatbot("Hello Chatbot"))