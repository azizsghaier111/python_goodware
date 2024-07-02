import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline, BertForMaskedLM, BertTokenizer
import pytorch_lightning as pl

# Function to initialize models and tokenizers.
def initialize_model_tokenizer(model_type, model_name):
    """
    Takes in a model type and model name, initialize the model and tokenizer and returns them.
    """
    if model_type == "seq2seq":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_type == "qa":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif model_type == "mlm":
        model = BertForMaskedLM.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    
    return tokenizer, model

# Translation
def translate(sequence, tokenizer, model):
    """
    Function for translating a given sequence of text.
    """
    inputs = tokenizer.encode(sequence, return_tensors='pt')
    outputs = model.generate(inputs, max_length=75, num_beams=5, early_stopping=True)
    translation = tokenizer.decode(outputs[0])
    return translation

# Question Answering
def get_answer(question, context, tokenizer, model):
    """
    Function for getting the answer to a given question within a provided context.
    """
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start, end = torch.argmax(start_scores), torch.argmax(end_scores)
    answer = tokenizer.convert_tokens_to_ids(inputs['input_ids'][0][start:end+1])
    return tokenizer.decode(answer)

# Masked Language Modeling
def fill_mask(masked_sentence, tokenizer, model):
    """
    Function for filling in the blank of a masked sentence.
    """
    input = torch.tensor([tokenizer.encode(masked_sentence)])
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    for token in top_5_tokens:
        print(masked_sentence.replace(tokenizer.mask_token, tokenizer.decode([token])))

if __name__ == '__main__':
    # Initialize the models and tokenizers
    translation_tokenizer, translation_model = \
        initialize_model_tokenizer("seq2seq", "Helsinki-NLP/opus-mt-fr-en")
    qa_tokenizer, qa_model = \
        initialize_model_tokenizer("qa", "bert-large-uncased-whole-word-masking-finetuned-squad")
    mlm_tokenizer, mlm_model = initialize_model_tokenizer("mlm", 'bert-base-uncased')

    # Simple test for translation
    sequence = 'Quel temps fait-il aujourd\'hui?'
    translation = translate(sequence, translation_tokenizer, translation_model)
    print(f'Translated Text: \n{translation}\n')

    # A test for question answering
    context = 'A transformer is a deep learning model that adopts the mechanism of self-attention,\
            pooling different positions of the input in its internal operation, allowing long distance interactions.'
    question = "What is a transformer?"
    answer = get_answer(question, context, qa_tokenizer, qa_model)
    print(f"Answer: \n{answer}\n")

    # A test for Masked Language Modeling
    masked_sentence = 'The quick brown [MASK] jumps over the lazy dog.'
    fill_mask(masked_sentence, mlm_tokenizer, mlm_model)