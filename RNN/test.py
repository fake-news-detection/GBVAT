# Import Packages
import pandas as pd
import spacy
import time
import re


'''
Text Preprocessing using spaCy 
1. Sentence detection / Tokenization
2. Stemming / Lemmatization
3. Stopwords
4. POS tagging
5. Punctuations and Noise removal
'''


class Spacy(object):

    def __init__(self):
        # python -m spacy download en_core_web_lg
        self.nlp = spacy.load("en_core_web_lg")
        pass

    def deNoise(self, text):
        text = re.sub(r'[“”""]', '', text) # removes quotes
        text = text.replace("'s", '')
        text = re.sub(r'[-]', ' ', text) # helps in splitting doc into sentences
        text = re.sub(r'http[\w:/\.]+', '', text) # removing urls
        text = re.sub(r'[^\.\w\s]', '', text) # removing everything but characters and punctuation
        text = re.sub(r'\.', '.', text) # replace periods with a single one
        text = re.sub(r'\s\s+', ' ', text) # replace multiple whitespace with one
        text = re.sub(r'\n', ' ', text) # removing line break
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text

    def stopWords(self, text):
        tokens = ""
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        text = text.split(" ")
        for word in text:
            if word not in spacy_stopwords:
                tokens = tokens + " " + word
        return tokens

    def lemmatize(self, tokens):
        lemma_token = ""
        tokens_object = self.nlp(tokens)
        for token in tokens_object:
            lemma_token = lemma_token + " " + token.lemma_
        return  lemma_token

    def set_custom_boundaries(self, doc):
        for token in doc[:-1]:
            if token.text == '--':
                doc[token.i+1].is_sent_start = True
        return doc

    def sentence_detect(self, text):
        self.nlp.add_pipe(self.set_custom_boundaries, before='parser')
        doc = self.nlp(text)
        sentences = list(doc.sents)
        for sentence in sentences:
            print(sentence)

    def tokenize(self, text):
        doc = self.nlp(text)
        print([token.text for token in doc])

class CleanCorpus(object):

    def main(self, document):
        S = Spacy()
        text = x[90]
        text = S.deNoise(text)
        tokens = S.stopWords(text)
        lemma_tokens = S.lemmatize(tokens)
        return lemma_tokens
        # S.sentence_detect(lemma_tokens)
        # S.tokenize(x[0])

if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv('GBVAT/data/processed_datasets/celebrityDataset.csv')

    # Feature Engineering
    df.nunique()
    df.isna().sum()
    df.Subject.fillna('', inplace=True)

    x = df.Subject + " " + df.Content
    y = pd.Series([0 if row == 'Fake' else 1 for row in df.Label])  # Series is 1D array but with same dtype

    CC = CleanCorpus()

    start = time.time()
    docs = [CC.main(row) for row in x]
    end = time.time()
    print("Cleaning the document took {} seconds".format(round(end - start)))


