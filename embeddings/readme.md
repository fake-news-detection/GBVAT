# Word Embeddings - A dense vector representation.

[![Word Embedding](https://adriancolyer.files.wordpress.com/2018/02/evolving-word-embeddings-fig-1.jpeg)]

Each word is represented by a point in the
high dimensional vector space and they are learnt and moved around based on the context of
words around it. Simply, word embeddings is allow words with similar meaning to be clustered together.
The dimension are chosen in an experimental way and hold abstract meanings. They have nothing to do with corpus size.


The current directory has codes for three traditional approaches of creating embeddings for a given corpus.

## Table of Contents
* [Word2Vec](#word2vec)
* [Glove](#glove)
* [Custom Embedding matrix](#custom)

### Word2vec:
It was developed by Google and has a core idea "Words will similar context tend to have similar meanings"
It has a shallow neural architecture, which tries to minimize its cost function while iterating over all texts with a fixed window size of context word.
In a general sense these algorithms look at fixed window of words for each target thus finding the context aka meaning it carries with itself.

The approach has two main algorithms to carry out the task
    a. CBOW: Continuous Bag of Words, where we are to predict the target word based on given context words.
    b. skip n-grams: Predict the context given a target word. **Preferred** due to better accuracy.

Word2vec requires a large amount of text, and performs optimization using **negative sampling and hierarchical softmax**.

Parameters for the constructor of class Word2Vec
*size*: no. of dimensions
*window*: contextwindow length each side
*min_count*: least frequency to be considered
*workers*: count of threads to train the model
*sg*: Algo choice 0 -> CBOW | 1 -> skip grams

[[Further Reading]](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html) :sparkles:

### glove

Insipired from word2vec, this model was developed by standford, which does not use a neural architecture. However, a co-occurance matrix of words in corpus is built and the resulting sparsity is condenced by a cost function that takes the difference between the product of two word vectors and log of the probability of their co-occurance.

The model is available across 4 variants : 50, 100, 200, 300 dimensions. More the dimension computational cost increases with more information being retained. It uses and explicit information for the vectorization and has proved better in performance in few datasets.

### Custom Embedding matrix

It is possible to train a custom embedding model. The code is present in the current directory.
First we need to get done with all text preprocessing steps and followed by declaring a embedding matrix of suitable dimnesions and intialize random values.
Use this weight matrix to pass through a neural layer and update the weights as each input  iterates of the entire corpus.
The Gensim Library could prove to be of great help here.
Similarity is determined by the cosine distance between two word vectors.

**Gensim**: Open Source Library in NLP for topic modeling. gensim provides a class Word2Vec to work with the below.
2. word2vec: Algorithm for learning Embeddings from a corpus of text.
