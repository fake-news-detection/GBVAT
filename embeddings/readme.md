# Word Embeddings - A dense vector representation.

[![Word Embedding](https://adriancolyer.files.wordpress.com/2018/02/evolving-word-embeddings-fig-1.jpeg)]

Each word is represented by a point in the
high dimensional vector space and they are learnt and moved around based on the context of
words around it. Simply, word embeddings is allow words with similar meaning to be clustered together.
The dimension are chosen in an experimental way and hold abstract meanings. They have nothing to do with corpus size.

The current directory has codes for three traditional approaches of creating embeddings for a given corpus.
1. **Word2vec:** It was developed by Google and has a core idea "Words will similar context tend to have similar meanings"
It has a shallow neural architecture, which tries to minimize its cost function while iterating over all texts with a fixed window size of context word.
In a general sense these algorithms look at fixed window of words for each target thus finding the context aka meaning it carries with itself.

The approach has two main algorithms to carry out the task


1. Gensim: Open Source Library in NLP for topic modeling. gensim provides a class Word2Vec to work with the below.
2. word2vec: Algorithm for learning Embeddings from a corpus of text.
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
