# GBVAT
This model uses raw embedding provided by Glove language model, a BiLSTM and regularization method of adversarial training for classification of corpus articles in a semi-supervised learning.


A word embedding is a dense vector representation. Each word is represented by a point in the
embedding space and they are learnt and moved around in the vector space based on the context of
word around it. Simply, WE is allows words with similar meaning to be clustered together.

1. Gensim: Open Source Library in NLP for topic modeling. gensim provides a class Word2Vec to work with the below.
2. word2vec: Algorithm for learning Embeddings from a corpus of text. In a general sense these algorithms
look at fixed window of words for each target thus finding the context aka meaning it carries with itself.
Word2vec requires a large amount of text.
    a. CBOW: Continuous Bag of Words
    b. skip n-grams:

    Parameters for the constructor of class Word2Vec
        1. size: no. of dimensions
        2. window:
        3. min_count: least frequency to be considered
        4. workers: count of threads to train the model
        5. sg: Algo choice 0 -> CBOW | 1 -> skip grams

3. Similarity is determined by the cosine distance between two word vectors.
