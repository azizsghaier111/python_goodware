import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class TextProcessor:
    def __init__(self):
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.sentiment_model = pipeline("sentiment-analysis")
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        
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
        

if __name__ == "__main__":
    processor = TextProcessor()
    print(processor.get_sentiment("I love using GPT-2."))
    print(processor.text_completion("Once upon a time, there was a little "))
    print(processor.translate("Hello, how are you?"))