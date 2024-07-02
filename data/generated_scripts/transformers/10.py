import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    pipeline, 
    BertForMaskedLM, 
    BertTokenizer
)
import pytorch_lightning as pl


def initialize_model_tokenizer(model_type, model_name):
    """
    Initialize and return the model and tokenizers based on model_type.
    """
    
    model = None
    tokenizer = None
    
    # check model type and initialize the model and tokenizer
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == "mlm":
        model = BertForMaskedLM.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

    return tokenizer, model


def translate(sequence, tokenizer, model):
    """
    Translate the input sequence and return the result.
    """
    inputs = tokenizer.encode(sequence, return_tensors='pt')
    outputs = model.generate(inputs, max_length=75, num_beams=5, early_stopping=True)
    translation = tokenizer.decode(outputs[0])
    
    return translation


def get_answer(question, context, tokenizer, model):
    """
    Get the answer to the question in the given context.
    """
    # encode inputs and pass to the model
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    outputs = model(**inputs)

    # get start and end scores of the answer from model outputs
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores)
    
    # convert predicted answer ids to tokens
    answer_ids = inputs['input_ids'][0][start:end+1]
    answer = tokenizer.convert_tokens_to_ids(answer_ids)
    
    return tokenizer.decode(answer)


def fill_mask(masked_sentence, tokenizer, model):
    """
    Predict the masked word in the sentence.
    """
    input = torch.tensor([tokenizer.encode(masked_sentence)])
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    for token in top_5_tokens:
        print(masked_sentence.replace(tokenizer.mask_token, tokenizer.decode([token])))


if __name__ == '__main__':
    # Translate model and tokenizer
    translation_model_name = "Helsinki-NLP/opus-mt-fr-en"
    translation_tokenizer, translation_model = \
        initialize_model_tokenizer("seq2seq", translation_model_name)

    # QA model and tokenizer
    qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    qa_tokenizer, qa_model = \
        initialize_model_tokenizer("qa", qa_model_name)

    # MLM model and tokenizer
    mlm_model_name = 'bert-base-uncased'
    mlm_tokenizer, mlm_model = initialize_model_tokenizer("mlm", mlm_model_name)

    # Perform a translation
    sequence = 'Quel temps fait-il aujourd\'hui?'
    translation = translate(sequence, translation_tokenizer, translation_model)
    print(f'Translated Text: \n{translation}\n')

    # Perform a QA
    context = 'A transformer is a deep learning model that adopts the mechanism of self-attention,\
            pooling different positions of the input in its internal operation, allowing long distance interactions.'
    question = "What is a transformer?"
    answer = get_answer(question, context, qa_tokenizer, qa_model)
    print(f"Answer: \n{answer}\n")

    # Fill in a mask
    masked_sentence = 'The quick brown [MASK] jumps over the lazy dog.'
    fill_mask(masked_sentence, mlm_tokenizer, mlm_model)