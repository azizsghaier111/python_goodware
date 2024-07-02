import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import (TFAutoModelForSequenceClassification,
                          AutoTokenizer, pipeline, 
                          TFGPT2LMHeadModel, GPT2Tokenizer)
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    
    class AIapplications:
        def __init__(self):
            self.model_classif = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
            self.conv_model = TFGPT2LMHeadModel.from_pretrained('gpt2')
            self.conv_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
        def classify_sequence(self, seq):
            inputs = self.tokenizer(seq, truncation=True, padding=True, return_tensors='tf')
            outputs = self.model_classif(inputs)[0]
            scores = outputs[0].numpy()
            scores = np.exp(scores) / np.sum(np.exp(scores))
            return {'positive': float(scores[1]), 'negative': float(scores[0])}
        
        def converse_ai(self, seq):
            inputs = self.conv_tokenizer.encode(seq + self.conv_tokenizer.eos_token, return_tensors='tf')
            reply_ids = self.conv_model.generate(inputs, max_length=1000, pad_token_id=self.conv_tokenizer.eos_token_id)
            return self.conv_tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        
        def sentiment_analysis(self, seq):
            inputs = self.tokenizer(seq, truncation=True, padding=True, return_tensors='tf')
            outputs = self.model_classif(inputs)[0].numpy()
            sentiment = np.argmax(outputs[0])
            return 'positive' if sentiment > 0 else 'negative'
        
        def tokenization(self, seq):
            return self.tokenizer.tokenize(seq)
    
    if __name__ == "__main__":
        
        ai = AIapplications()
        
        # Print the tokenization
        print('Tokenization: \n', ai.tokenization('Hello, how are you?'))
        print('\n\n\n')
        
        # Print the conversation ai
        print('Conversation bot says: \n', ai.converse_ai('Hello'))
        print('\n\n\n')
        
        # Print document classification
        print("Classification Score: \n", ai.classify_sequence("I am so happy."))
        print("Sentiment Analysis Result: \n", ai.sentiment_analysis('I am so happy.'))
        print('\n\n\n')

except Exception as e:
    print(f"An error occurs during loading the models or during their usage: {e}")