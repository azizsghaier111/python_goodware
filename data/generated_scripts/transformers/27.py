import torch
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForTokenClassification, AutoModelWithLMHead

class NLPtasks:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelWithLMHead.from_pretrained("distilbert-base-uncased")
        self.lang_detect_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def tokenize(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        return inputs

    def text_generation(self, texts):
        inputs = self.tokenize(texts)
        outputs = self.model.generate(**inputs, max_length=200, temperature=1.0, do_sample=True)
        return [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

    def named_entity_recognition(self, texts):
        tokens = self.tokenize(texts)
        outputs = self.model(**tokens)
        token_labels_indices = torch.argmax(outputs.logits, dim=-1)
        return [self.tokenizer.decode(token_labels_index) for token_labels_index in token_labels_indices]

    def question_answering(self, contexts, questions):
        inputs = self.qa_tokenizer(questions, contexts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.qa_model(**inputs)
        answer_starts = torch.argmax(outputs.start_logits, dim=-1)
        answer_ends = torch.argmax(outputs.end_logits, dim=-1)
        return [self.qa_tokenizer.decode(inputs.input_ids[i, answer_starts[i]:answer_ends[i] + 1]) for i in range(len(contexts))]

    def language_detection(self, text):
        return detect(text)

if __name__ == "__main__":
    nlp_tasks = NLPtasks()

    texts = ["Hello, my name is John Doe and I work at Google.", "How are you today?"]
    print(nlp_tasks.text_generation(texts))

    ner_texts = ["John Doe works at Google in California.", "The capital of France is Paris."]
    print(nlp_tasks.named_entity_recognition(ner_texts))

    contexts = ["I had pizza and pasta for dinner.", "My name is John Doe."]

    questions = ["What did I have for dinner?", "What is my name?"]
    print(nlp_tasks.question_answering(contexts, questions))

    lang_texts = ["Hello, how are you?", "Bonjour, comment ça va?"]
    for text in lang_texts:
        print(nlp_tasks.language_detection(text))