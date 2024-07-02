import torch
import numpy as np
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer,
                          RobertaForSequenceClassification, RobertaTokenizer, BartForConditionalGeneration, BartTokenizer)

# Initialising the GPT-2 tokenizer and model
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize the Named entity recognition pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize the translation model and tokenizer (T5)
translation_model = T5ForConditionalGeneration.from_pretrained('t5-base')
translation_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Initialize the text simplification model and tokenizer (BART)
simp_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
simp_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Initialize sequence classification model and tokenizer (RoBERTa)
classif_model = RobertaForSequenceClassification.from_pretrained('roberta-base')
classif_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# Text Generation
def generate_text(seed_text, max_length=75):
    input_text = tokenizer_gpt2.encode(seed_text, return_tensors="pt")
    with torch.no_grad():
        output_text = model_gpt2.generate(input_text, max_length=max_length, num_return_sequences=1)
    return tokenizer_gpt2.decode(output_text[0])

# Named entity recognition (NER)
def named_entity_recognition(text):
    return ner_pipeline(text)

# Text Translation 
def translate(text, target_language="en"):
    prep_text = "translate English to " + target_language + ": " + text
    input_ids = translation_tokenizer.encode(prep_text, return_tensors="pt")
    with torch.no_grad():
        decoded_ids = translation_model.generate(input_ids, max_length=120, num_beams=4, early_stopping=True)
    return translation_tokenizer.decode(decoded_ids[0])

# Text Simplification
def simplify(text):
    inputs = simp_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
    outputs = simp_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return simp_tokenizer.decode(outputs[0])

# Sequence Classification
def sequence_classification(text):
    inputs = classif_tokenizer.encode(text, return_tensors="pt")
    outputs = classif_model(inputs)[0]
    predictions = torch.nn.functional.softmax(outputs, dim=-1)
    return predictions

if __name__ == '__main__':
    seed_text = "The sky is "

    # Text Generation
    generated_text = generate_text(seed_text)
    print("Generated text: ", generated_text)

    # NER
    text = "Apple is looking at buying a U.K. startup for $1 billion"
    ner_results = named_entity_recognition(text)
    print("NER Results: ", ner_results)

    # Text Translation
    translated_text = translate("Le ciel est bleu", target_language="en")
    print("Translated text: ", translated_text)

    # Text Simplification
    simpl_text = simplify("Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells consumer electronics, computer software, and online services.")
    print("Simplified text: ", simpl_text)

    # Sequence Classification
    classif_result = sequence_classification("Apple is looking at buying a U.K. startup for $1 billion")
    print("Classification result: ", classif_result)