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

# Shuffle data random before splitting
import random
indices = np.arange(sequences.shape[0])
random.Random(1).shuffle(indices)
data = sequences[indices]
labels = y[indices]

# Word Embeddings : the dimension are chosen in a experimental way have abstract meanings. They have nothing to do with corpus size.
# larger dimension will capture more information but harder to use.

embeddings_index = {}
f = open(GLOVE_DIR, encoding='utf-8')
print('Loading Glove from: ', GLOVE_DIR, '...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print('Found %s word vectors.' % len(embeddings_index))
print('\nDone.\nProcedding with Embedded Matrix...', end='')

#Create an embedding matrix
# first create a matrix of zeros, this is our embedding matrix
embeddings_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
#embeddings_matrix = np.random.random(((20568),EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embeddings_matrix[i] = np.random.random(EMBEDDING_DIM)
print('\nCompleted')