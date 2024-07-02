import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline, GPT2Tokenizer, TFGPT2LMHeadModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    # Load pre-trained model for sequence classification
    model_classif = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    # For sequence classification
    def classify_sequence(seq, model=model_classif, tokenizer=tokenizer):
        inputs = tokenizer(seq, return_tensors='tf')
        outputs = model(inputs)
        scores = outputs[0][0].numpy()
        scores = np.exp(scores) / np.sum(np.exp(scores))
        return {'positive': float(scores[1]), 'negative': float(scores[0])}

    # Load pre-trained model for conversational AI
    conv_model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    conv_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # For conversational AI
    def converse_ai(seq, model=conv_model, tokenizer=conv_tokenizer):
        inputs = tokenizer.encode(seq + tokenizer.eos_token, return_tensors='tf')
        reply_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Load pre-trained model for sentiment analysis
    sentiment_model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    sentiment_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    # For sentiment analysis
    def sentiment_analysis(seq, model=sentiment_model, tokenizer=sentiment_tokenizer):
        inputs = tokenizer(seq, return_tensors='tf')
        outputs = model(inputs)[0].numpy()
        sentiment = np.argmax(outputs[0])
        return 'positive' if sentiment > 0 else 'negative'

except Exception as e:
    print(f"An error occurs during loading the models or during their usage: {e}")