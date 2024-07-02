import torch
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, pipeline,
                            T5ForConditionalGeneration, T5Tokenizer,
                            BartForConditionalGeneration, BartTokenizer)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize the Named Entity Recognition (NER) pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize the translation model and tokenizer
translation_model = T5ForConditionalGeneration.from_pretrained('t5-base')
translation_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Initialize the summarization model and tokenizer
summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
summarization_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def generate_text(seed_text, max_length=75):
    input_text = torch.tensor([tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output_text[0])

def named_entity_recognition(text):
    return ner_pipeline(text)

def translate(text, target_language="en"):
    input_ids = translation_tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        decoded_ids = translation_model.generate(input_ids, max_length=150, num_beams=2, early_stopping=True)
    return translation_tokenizer.decode(decoded_ids[0])

def summarize(text):
    inputs = summarization_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = summarization_model.generate(inputs, max_length=150, min_length=50)
    return summarization_tokenizer.decode(outputs[0])

if __name__ == '__main__':
    seed_text = "The sky is "
    generated_text = generate_text(seed_text)
    print("\nGenerated text: ", generated_text)

    text = "Apple is looking at buying a U.K. startup for $1 billion"
    ner_results = named_entity_recognition(text)
    print("\nNER Results: ", ner_results)

    conversation = generate_text("Hello, how are you?")
    print("\nAI conversation: ", conversation)

    translated_text = translate("Le ciel est bleu", target_language="en")
    print("\nTranslated text: ", translated_text)

    summary_text = "Wikipedia is a free online encyclopedia, created and edited by volunteers around the world and hosted by the Wikimedia Foundation."
    summary_results = summarize(summary_text)
    print("\nSummary of the text: ", summary_results)

    # Repeat the AI functions

    for i in range(4): # Explain something in-depth, you might alter this to suit your need
        seed_text = "Tell me more about AI"
        generated_text = generate_text(seed_text)
        print("\nGenerated text: ", generated_text)

        text = "Google is planning to buy another AI company"
        ner_results = named_entity_recognition(text)
        print("\nNER Results: ", ner_results)

        conversation = generate_text("Tell me more about Machine Learning!")
        print("\nAI conversation: ", conversation)

        translated_text = translate("L'IA est l'avenir", target_language="en")
        print("\nTranslated text: ", translated_text)

        summary_text = "Artificial Intelligence (AI) is becoming an integral part of our daily lives. More and more industries are integrating AI to improve their efficiency and productivity."
        summary_results = summarize(summary_text)
        print("\nSummary of the text: ", summary_results)