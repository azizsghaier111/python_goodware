import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, BertForMaskedLM, BertTokenizer
from transformers import TextClassificationPipeline, TFAutoModelForSequenceClassification
import tensorflow as tf
import pytorch_lightning as pl

# Initializing the tokenizer and model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# For sentiment analysis
sentiment_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_tokenizer = GPT2Tokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipeline = TextClassificationPipeline(model=sentiment_model, tokenizer=sentiment_tokenizer)

# For MLM
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
chatbot = pipeline('text-generation', model=mlm_model, tokenizer=mlm_tokenizer)

def analyze_sentiment(text):
    """
    Function for sentiment analysis
    """
    result = sentiment_pipeline(text)
    return result

def masked_language_modeling(text):
    """
    Function for masked language modeling
    """
    result = chatbot(text)
    return result

def tokenize_text(sentence):
    """
    Function for tokenization
    """
    tokens = gpt2_tokenizer.tokenize(sentence)
    return tokens

def convert_token_to_id(tokens):
    """
    Function for converting tokens to ids
    """
    token_ids = gpt2_tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

def convert_id_to_tokens(token_ids):
    """
    Function for converting token ids to tokens
    """
    tokens = gpt2_tokenizer.convert_ids_to_tokens(token_ids)
    return tokens

def generate_text(seed_text, max_length=75):
    """
    Function for text generation
    """
    input_text = torch.tensor([gpt2_tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = gpt2_model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return gpt2_tokenizer.decode(output_text[0])

def text_completion(seed_text, k=50):
    """
    Function for text completion
    """
    input_text = torch.tensor([gpt2_tokenizer.encode(seed_text)])
    with torch.no_grad():
        outputs = gpt2_model(input_text)
        predictions = outputs[0][0]

    predicted_index = torch.topk(predictions, k)
    predicted_text = gpt2_tokenizer.decode(predicted_index.indices)

    return predicted_text

if __name__ == '__main__':
    seed_text = "The sky is "

    # Tokenization
    tokens = tokenize_text(seed_text)
    print("Tokens from text: ", tokens)
    token_ids = convert_token_to_id(tokens)
    print("Token Ids: ", token_ids)
    tokens = convert_id_to_tokens(token_ids)
    print("Tokens from Ids: ", tokens)

    # Text Generation
    generated_text = generate_text(seed_text)
    print("Generated text: ", generated_text)

    # Text Completion
    completed_text = text_completion(seed_text)
    print("Completed text: ", completed_text)

    # Sentiment Analysis Example
    sentiment = analyze_sentiment(seed_text)
    print("Sentiment Analysis: ", sentiment)

    # Masked Language Model Example
    mlm_result = masked_language_modeling("The sky is [MASK].")
    print("Masked Language Model: ", mlm_result)