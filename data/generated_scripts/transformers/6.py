import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    pipeline, 
    BertForMaskedLM, 
    BertTokenizer, 
    TextClassificationPipeline, 
    TFAutoModelForSequenceClassification, 
    AutoTokenizer, 
    pipeline,
    AutoModelForSeq2SeqLM
)
import tensorflow as tf
import pytorch_lightning as pl
from langdetect import detect

# Initializing different models and their corresponding tokenizers
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

question_answering_model= AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl")
question_answering_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
question_answering_pipeline = pipeline('question-answering', model=question_answering_model, tokenizer=question_answering_tokenizer)

sentiment_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipeline = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer)

mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mlm_pipeline = pipeline('fill-mask', model=mlm_model, tokenizer=mlm_tokenizer)

def detect_language(text):
    language = detect(text)
    return language

def answer_question(question, context):
    result = question_answering_pipeline(question=question, context=context)
    return result

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result

def masked_language_modeling(text):
    result = mlm_pipeline(text)
    return result

def tokenize_text(sentence):
    tokens = gpt2_tokenizer.tokenize(sentence)
    return tokens

def convert_token_to_id(tokens):
    token_ids = gpt2_tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

def convert_id_to_tokens(token_ids):
    tokens = gpt2_tokenizer.convert_ids_to_tokens(token_ids)
    return tokens

def generate_text(seed_text, max_length=75):
    input_text = torch.tensor([gpt2_tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = gpt2_model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return gpt2_tokenizer.decode(output_text[0])

if __name__ == '__main__':
    seed_text = "The sky is "
    context = "The sky's color during a clear daytime is blue due to Rayleigh scattering of sunlight."

    # Language Detection
    language = detect_language(seed_text)
    print("Language Detection: ", language)

    # Question Answering
    question = "Why is the sky blue?"
    answer = answer_question(question, context)
    print("Question Answering: ", answer)

    # Sentiment Analysis
    sentiment = analyze_sentiment(seed_text)
    print("Sentiment Analysis: ", sentiment)

    # Masked Language Model
    mlm_result = masked_language_modeling("The sky is [MASK].")
    print("Masked Language Model: ", mlm_result)

    # Text Generation
    generated_text = generate_text(seed_text)
    print("Generated text: ", generated_text)