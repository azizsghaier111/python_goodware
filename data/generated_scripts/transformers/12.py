import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer

def create_directory_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_model_and_tokenizer():
    # Initialising the tokenizer and model
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Named entity recognition pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    # Translation model and tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    return gpt2_tokenizer, gpt2_model, ner_pipeline, t5_model, t5_tokenizer

# Model and tokenizer path
path = './models'

# Text Generation
def generate_text(tokenizer, model, seed_text, max_length=75):
    input_text = torch.tensor([tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output_text[0])

# Named entity recognition
def named_entity_recognition(ner_pipeline, text):
    return ner_pipeline(text)

# Translation 
def translate(tokenizer, model, text, target_language="en"):
    input_ids = tokenizer.encode("translate English to French: " + text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        translated_ids = model.generate(input_ids, max_length=75, num_beams=2, early_stopping=True)
    return tokenizer.decode(translated_ids[0])

def main():
    # Create directory path
    create_directory_path(path)
    
    # Download models and tokenizers
    gpt2_tokenizer, gpt2_model, ner_pipeline, t5_model, t5_tokenizer = download_model_and_tokenizer()
    
    # Print models and tokenizers
    print(gpt2_tokenizer)
    print(gpt2_model)
    print(ner_pipeline)
    print(t5_model)
    print(t5_tokenizer)
    
    seed_text = "The sky is "

    # Text Generation
    generated_text = generate_text(gpt2_tokenizer, gpt2_model, seed_text)
    print("\nGenerated text: ", generated_text)

    # Named entity recognition
    text = "Apple is looking at buying a U.K. startup for $1 billion"
    ner_results = named_entity_recognition(ner_pipeline, text)
    print("\nNER Results: ", ner_results)

    # Text Translation
    translated_text = translate(t5_tokenizer, t5_model, "The sky is blue", target_language="fr")
    print("\nTranslated text: ", translated_text)

if __name__ == '__main__':
    main()