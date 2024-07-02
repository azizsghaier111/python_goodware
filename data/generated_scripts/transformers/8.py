import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer

# Initialising the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Named entity recognition pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Translation model and tokenizer
translation_model = T5ForConditionalGeneration.from_pretrained('t5-base')
translation_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Text Generation
def generate_text(seed_text, max_length=75):
    input_text = torch.tensor([tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output_text[0])

# Named entity recognition
def named_entity_recognition(text):
    return ner_pipeline(text)

# Conversational AI - in this case simply use the text generation function

# Translation 
def translate(text, target_language="en"):
    input_ids = translation_tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        decoded_ids = translation_model.generate(input_ids, max_length=75, num_beams=2, early_stopping=True)
    return translation_tokenizer.decode(decoded_ids[0])

if __name__ == '__main__':
    seed_text = "The sky is "

    # Text Generation
    generated_text = generate_text(seed_text)
    print("Generated text: ", generated_text)

    # Named entity recognition
    text = "Apple is looking at buying a U.K. startup for $1 billion"
    ner_results = named_entity_recognition(text)
    print("NER Results: ", ner_results)

    # Converse with AI
    conversation = generate_text("Hello, how are you?")
    print("AI conversation: ", conversation)

    # Text Translation
    translated_text = translate("Le ciel est bleu", target_language="en")
    print("Translated text: ", translated_text)