import re
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


df = pd.read_csv('GBVAT/data/processed_datasets/celebrityDataset.csv')
# Extract relevant features
df.nunique()
df.isna().sum()
df['Subject'].fillna('',inplace=True) # Replace all missing values
x = df['Subject'] + " " + df['Content']
#y = pd.get_dummies(df['Label'])
y = [1 if row == 'Fake' else 0 for row in df['Label']]
y = np.array(y) # Dummy Encoding

# Clean the texts
def clean_text(text, remove_stopwords=True):
    output = ""
    text = str(text).replace(r'http[\w:/\.]+', '') # removing urls
    text = str(text).replace(r'[^\.\w\s]', '') # removing everything but characters and punctuation
    text = str(text).replace(r'\.\.+', '.') # replace multiple periods with a single one
    text = str(text).replace(r'\.', ' . ') # replace periods with a single one
    text = str(text).replace(r'\s\s++', ' ') # replace multiple whitespace with one
    text = str(text).replace(r'\n', '') # removing line break
    text = re.sub(r'[^\w\s]', '', text.lower()) # lower texts
    if remove_stopwords:
        text = text.split(" ")
        for word in text:
            if word not in stopwords.words('english'):
                output = output + " " + word
    return output

# Clean the corpus
start = time.time()
docs = [clean_text(row) for row in x]
end = time.time()
print("Cleaning the document took {} seconds".format(round(end - start)))

# Parameters
MAX_VOCAB_SIZE = 1000000 # maximum no of unique words
MAX_DOC_LENGTH = 500 # maximum no of words in each sentence
EMBEDDING_DIM = 300 # Embeddings dimension from Glove directory
GLOVE_DIR = 'models/glove.6B/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'
TEST_SPLIT = 0.2

# Tokenize & pad sequences
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(docs)
encoded_docs = tokenizer.texts_to_sequences(docs)
word_index = tokenizer.word_index
print('Vocabulary size :', len(word_index))
sequences = pad_sequences(encoded_docs, padding='post', maxlen=MAX_DOC_LENGTH)
print('Shape of data tensor:', sequences.shape)
print('Shape of label tensor', y.shape)

d = word_index.items()
n = list(d)
