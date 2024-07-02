import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Initializing the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def tokenize_text(sentence):
    """
    Function for tokenization
    """
    tokens = tokenizer.tokenize(sentence)
    return tokens

def convert_token_to_id(tokens):
    """
    Function for converting tokens to ids
    """
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

def convert_id_to_tokens(token_ids):
    """
    Function for converting token ids to tokens
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokens

def generate_text(seed_text, max_length=75):
    """
    Function for text generation
    """
    input_text = torch.tensor([tokenizer.encode(seed_text)])
    with torch.no_grad():
        output_text = model.generate(input_text, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output_text[0])

def text_completion(seed_text, k=50):
    """
    Function for text completion
    """
    input_text = torch.tensor([tokenizer.encode(seed_text)])
    with torch.no_grad():
        outputs = model(input_text)
        predictions = outputs[0][0]

    predicted_index = torch.topk(predictions, k)
    predicted_text = tokenizer.decode(predicted_index.indices)

    return predicted_text

# Sentiment analysis pipeline
def get_sentiment(text):
    sentiment_analysis = pipeline("sentiment-analysis")
    result = sentiment_analysis(text)[0]
    return f"label: {result['label']}, with score: {round(result['score'], 4)}"

# Named Entity Recognition pipeline
def get_entities(text):
    ner_pipe = pipeline("ner", grouped_entities=True)
    results = ner_pipe(text)
    return results

# Text simplification using T5
def simplify_text(text):
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

    inputs = t5_tokenizer.encode("simplify: " + text, return_tensors="pt", max_length=512)
    outputs = t5_model.generate(inputs, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0])

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

    # Sentiment Analysis
    sentiment = get_sentiment(generated_text)
    print("Sentiment: ", sentiment)

    # Named Entity Recognition
    entities = get_entities(seed_text)
    print("Entities: ", entities)

    # Text Simplification
    simplified_text = simplify_text(seed_text)
    print("Simplified text: ", simplified_text)