import torch
from transformers import (
    AutoModelForSequenceClassification,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime

# Load BertForMaskedLM Model
model_mask = "bert-base-uncased"
tokenizer_mask = BertTokenizer.from_pretrained(model_mask)
model_mask = BertForMaskedLM.from_pretrained(model_mask)

# Feature Extraction
model_feat = AutoModelForSequenceClassification.from_pretrained(model_mask, num_labels = 2, output_hidden_states = True)
hidden_states = model_feat(input_id)[2]

# Load BertForQuestionAnswering Model
model_que = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Prepare Masked Text for entity extraction
def prepare_masked_text(text):
    tokenized_text = tokenizer_mask.tokenize(text.lower())
    mask_index =  tokenized_text.index('[mask]')
    tokenized_text[mask_index] = tokenizer_mask.mask_token
    return tokenized_text

# Entity filling
def predict_masked_entity(masked_text, model):
    index = masked_text.index("[MASK]")
    input_ids = tokenizer_mask.encode(masked_text, return_tensors='pt')
    with torch.no_grad():
        predict = model(input_ids)[0]
    predicted_index = torch.argmax(predict[0, index]).item()
    predicted_token = tokenizer_mask.convert_ids_to_tokens([predicted_index])[0]
    return predicted_token

# Question answering
def answer_question(question, answer_text):
    input_text = "[CLS] " + question + " [SEP] " + answer_text + " [SEP]"
    input_ids = tokenizer_que.encode(input_text)
    segment_ids = [0] * len(input_ids)
    start_scores, end_scores = model_que(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer_que.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return answer

#print feature
print("Hidden States : ", hidden_states[-1][0])
# Entity masking
masked_text = prepare_masked_text("The capital of France is [MASK].")
predicted_entity = predict_masked_entity(masked_text, model_mask)
print("Predicted Entity: ", predicted_entity)
# Question Answering
answer_text = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
question = "What is the Eiffel Tower?"
answer = answer_question(question, answer_text)
print("Question: ", question)
print("Answer: ", answer)