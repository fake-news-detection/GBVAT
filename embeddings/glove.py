from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot

# define training data
doc1 = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

doc2 = [['the', 'children', 'play', 'in', 'the', 'work'],
            ['The', 'play', 'is', 'in', 'old theatre']]

# train model
model = Word2Vec(doc1, min_count=1)
print(model) # summarize the loaded model
words = list(model.wv.vocab) # summarize vocabulary
print(words)
text = model['sentence']
print(text) # access vector for one word

# save model
model.wv.save_word2vec_format('GBVAT/embeddings/test_model.txt', binary=False)
model.wv.save_word2vec_format('GBVAT/embeddings/test_model.bin') # both behave similar
model.save('GBVAT/embeddings/test_model.bin')
# load model
new_model = Word2Vec.load('GBVAT/embeddings/test_model.bin')
print(new_model)

# Visualize word embeddings
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# Scatterplot
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0]+.0005, result[i, 1]+.0005))
pyplot.show()