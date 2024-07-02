import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5Tokenizer, T5ForConditionalGeneration, pipeline

# Initializing the tokenizer and model
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

# Initializing the tokenizer and model for text simplification (T5)
tokenizer_t5 = T5Tokenizer.from_pretrained('google/t5-v1_1-large')
model_t5 = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-large')

# Initializing mask-filling
mask_filling = pipeline('fill-mask', model='distilroberta-base', tokenizer='distilroberta-base')

def mask_fill(text:str)->str:
    """
    Function for mask filling
    """
    return mask_filling(text)

def tokenize_text(sentence:str)->list:
    """
    Function for tokenization
    """
    tokens = tokenizer_gpt2.tokenize(sentence)
    return tokens

def convert_token_to_id(tokens:list)->list:
    """
    Function for converting tokens to ids
    """
    token_ids = tokenizer_gpt2.convert_tokens_to_ids(tokens)
    return token_ids

def convert_id_to_tokens(token_ids:list)->list:
    """
    Function for converting token ids to tokens
    """
    tokens = tokenizer_gpt2.convert_ids_to_tokens(token_ids)
    return tokens

def generate_text(seed_text:str, max_length:int=200)->str:
    """
    Function for text generation
    """
    input_text = torch.tensor([tokenizer_gpt2.encode(seed_text)])
    with torch.no_grad():
        output_text = model_gpt2.generate(input_text, max_length=max_length, num_return_sequences=1)
    return tokenizer_gpt2.decode(output_text[0])

def text_simplification(text:str)->str:
    """
    Function for text simplification using T5 
    """
    input_text = tokenizer_t5.encode('simplify: ' + text, return_tensors='pt')
    with torch.no_grad():
        output_text = model_t5.generate(input_text, max_length=200, num_return_sequences=1)
    return tokenizer_t5.decode(output_text[0])

if __name__ == '__main__':
    seed_text = "The sky is blue but the earth is round"

    # Tokenization
    tokens = tokenize_text(seed_text)
    print("Tokens from text: ", tokens)

    # Tokens to Ids
    token_ids = convert_token_to_id(tokens)
    print("Token Ids: ", token_ids)
    
    # Ids to Tokens
    tokens = convert_id_to_tokens(token_ids)
    print("Tokens from Ids: ", tokens)

    # Text Generation
    generated_text = generate_text(seed_text)
    print("Generated text: ", generated_text)
    
    # Text Simplification 
    simplification_text = text_simplification(seed_text)
    print("Simplified text: ", simplification_text)
    
    # Mask Filling
    mask_filled_text = mask_fill(seed_text.replace("blue", "<mask>"))
    print("Mask filled text: ", mask_filled_text)