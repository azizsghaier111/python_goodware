import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline, GPT2Tokenizer, TFGPT2LMHeadModel 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BertForQuestionAnswering, BertTokenizer
from symspellpy import SymSpell, Verbosity

def load_model():
    try:
        #Load pre-trained model for sequence classification
        model_classif = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        tokenizer_classif = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        
        #Load pretrained conversational model
        conv_model = TFGPT2LMHeadModel.from_pretrained('gpt2')
        conv_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Load pretrained models for Text simplification and question answering
        simpl_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        simpl_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        # SymSpell for spell checking
        sym_spell = SymSpell(max_dictionary_edit_distance = 2, prefix_length = 7)
        dictionary_path = sym_spell.pd_dictionary_path
        sym_spell.load_dictionary(dictionary_path, term_index = 0, count_index = 1)

        return {'model_classif': model_classif,'tokenizer_classif': tokenizer_classif,
                'conv_model': conv_model,'conv_tokenizer': conv_tokenizer,
                'simpl_model': simpl_model,'simpl_tokenizer': simpl_tokenizer,
                'qa_model': qa_model, 'qa_tokenizer': qa_tokenizer,
                'sym_spell': sym_spell}

    except Exception as e:
        print(f"Error occurred during loading the models: {e}")

def spell_check(text, model):
    sym_spell = model['sym_spell']
    suggestions = sym_spell.lookup_compound(text, max_edit_distance = 2)
    return suggestions[0].term

def simplify_text(text, model):
    tokenizer = model['simpl_tokenizer']
    model = model['simpl_model']
    inputs = tokenizer.encode("summarize: "+text, return_tensors='pt', max_length = 1024, truncation = True)
    outputs = model.generate(inputs, max_length = 250, min_length = 40, length_penalty = 2.0, num_beams = 4, early_stopping = True)
    return tokenizer.decode(outputs[0])

def answer_question(question, context, model):
    tokenizer = model['qa_tokenizer']
    model = model['qa_model']
    input_text = "[CLS] " + question + " [SEP] " + context + " [SEP]"
    input_ids = tokenizer.encode(input_text)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    outputs = model(torch.tensor([input_ids]).to(torch.device('cpu')), token_type_ids = torch.tensor([token_type_ids]).to(torch.device('cpu')))
    return tokenizer.decode(input_ids[torch.argmax(outputs.start_logits) : torch.argmax(outputs.end_logits)+1])

if __name__ == '__main__':
    model = load_model()
    spell_checked = spell_check("speling chekcer", model)
    simplified = simplify_text("He was quite surprised at the unexpected turn of events.", model)
    answer = answer_question('What is deep learning?','Deep learning is a subfield of machine learning.', model)
    print(f"Spell Check: {spell_checked}\nSimplified Text: {simplified}\nAnswer: {answer}")