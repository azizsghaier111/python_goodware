import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialising the tokenizer and model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Named entity recognition pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Translation model and tokenizer
translation_model = T5ForConditionalGeneration.from_pretrained('t5-base')
translation_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Simplification model and tokenizer
simplification_tokenizer = AutoTokenizer.from_pretrained('t5-base')
simplification_model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

def generate_text_precise(seed_text, max_length=75):
    input_text = torch.tensor([gpt2_tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = gpt2_model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return gpt2_tokenizer.decode(output_text[0])

def named_entity_recognition_precise(input_txt):
    return ner_pipeline(input_txt)

def translate_precise(input_txt, target_lang="en"):
    input_ids = translation_tokenizer.encode(input_txt, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        decoded_ids = translation_model.generate(input_ids, max_length=75, num_beams=2, early_stopping=True)
    return translation_tokenizer.decode(decoded_ids[0])

def tokenize_precise(input_txt):
    return gpt2_tokenizer.encode(input_txt)

def simplify_text_precise(input_txt, max_length=75):
    inputs = simplification_tokenizer.encode("simplify: "+input_txt, return_tensors="pt", max_length=max_length)   
    outputs = simplification_model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return simplification_tokenizer.decode(outputs[0])

if __name__ == '__main__':
    seed_txt = "The sky is "

    # Text Generation
    generated_txt = generate_text_precise(seed_txt)
    print("Generated Text: ", generated_txt)

    # Named entity recognition
    txt = "Apple is looking at buying a U.K. startup for $1 billion"
    ner_results = named_entity_recognition_precise(txt)
    print("NER Result: ", ner_results)

    # Text translation
    translated_txt = translate_precise("Le ciel est bleu", target_lang="en")
    print("Translated Text: ", translated_txt)

    # Tokenization
    print("Tokenized Text: ", tokenize_precise("Tokenize this text."))

    # Text Simplification
    print("Simplified Text: ", simplify_text_precise("It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."))