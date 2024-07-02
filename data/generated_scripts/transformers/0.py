# Some necessary imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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