import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import pytorch_lightning as pl

# Initializing the tokenizer, model and pipeline
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
simplification_pipeline = pipeline(task='text2text-generation', model='google/t5-v1_1-large')

# NER pipeline initialization
ner_pipeline = pipeline('ner')

# Language detection pipeline
lang_det = pipeline('feature-extraction')

def generate_text(seed_text, max_length=200):
    input_text = torch.tensor([tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output_text[0])

def lang_detect(text):
    features = lang_det(text)[0]
    if sum(features) > 0:
        return 'English' # can integrate any model for multiple language detection
    else:
        return 'Not a language'

def named_entity_recognition(text):
    result = ner_pipeline(text)
    return [(entity['word'], entity['entity_group']) for entity in result]

if __name__ == '__main__':
    seed_text = "The sky is blue and Google was founded in September 1998."

    # Text Generation
    print("Generated text: ", generate_text(seed_text))

    # Language Detection
    print("Detected language: ", lang_detect(seed_text))

    # Named Entity Recognition
    print("Named Entities: ", named_entity_recognition(seed_text))