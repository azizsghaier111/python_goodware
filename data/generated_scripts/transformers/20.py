import torch
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, BertForMaskedLM, BertTokenizer

# Initializing the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# For NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# For MLM
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mlm_pipeline = pipeline('fill-mask', model=mlm_model, tokenizer=mlm_tokenizer)

def named_entity_recognition(text):
    """
    Function for Named Entity Recognition
    """
    result = ner_pipeline(text)
    return result

def masked_language_modeling(text):
    """
    Function for Masked Language Modeling
    """
    result = mlm_pipeline(text)
    return result

if __name__ == "__main__":
    text = "Hugging Face is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."

    # Named Entity Recognition Example
    ner_result = named_entity_recognition(text)
    print("Named Entity Recognition: ", ner_result)

    # Masked Language Model Example
    mlm_result = masked_language_modeling("Hugging Face is a company based in [MASK].")
    print("Masked Language Model: ", mlm_result)