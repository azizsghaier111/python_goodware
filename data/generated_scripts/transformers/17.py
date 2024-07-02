import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, T5Tokenizer, T5ForConditionalGeneration

class TextProcessor:
    def __init__(self):
        self.tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
        self.model_gpt2 = AutoModelWithLMHead.from_pretrained("gpt2")

        self.tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')
        self.model_t5 = T5ForConditionalGeneration.from_pretrained('t5-small')

        self.summarization_pipeline = pipeline('summarization')
        self.translation_pipeline = pipeline('translation_en_to_de')

    def tokenize(self, text):
        return self.tokenizer_gpt2.tokenize(text)

    def text_completion(self, text):
        input_tokens = self.tokenizer_gpt2.encode(text, return_tensors='pt')
        output = self.model_gpt2.generate(input_tokens, do_sample=True, 
                                           max_length=50, 
                                           pad_token_id=self.tokenizer_gpt2.eos_token_id)

        return self.tokenizer_gpt2.decode(output[:, input_tokens.shape[-1]:][0], skip_special_tokens=True)

    def translate_to_german(self, text):
        result = self.translation_pipeline(text)
        return result[0]['translation_text']

    def summarize(self, text):
        return self.summarization_pipeline(text)

    def make_conversation(self, text):
        encoded = self.tokenizer_t5.encode(text + '</s>', return_tensors='pt')
        output = self.model_t5.generate(encoded, max_length=1000, num_beams=4, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
        return self.tokenizer_t5.decode(output[:, 0:][0], skip_special_tokens=True)

processor = TextProcessor()

print("Tokenize: ", processor.tokenize("Once upon a time, there was a little "))
print("Text Completion: ", processor.text_completion("Once upon a time, there was a little "))
print("Translation to German: ", processor.translate_to_german("Once upon a time, there was a little "))
print("Summary: ", processor.summarize("Once upon a time in a kingdom"))
print("Make conversation: ", processor.make_conversation("Hello, How are you?"))