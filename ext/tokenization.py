"""For tokenization of text."""
import spacy


class SpaCyTokenizer:

    def __init__(self):
        self.nlp = spacy.load('en')

    def __call__(self, text):
        doc = self.nlp(text)
        return [tok.text for tok in doc]
