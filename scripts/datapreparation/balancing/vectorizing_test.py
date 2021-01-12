
from gensim.models import word2vec
from pymongo import database
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from matplotlib import pyplot

import scripts.mongoConnection as mc

##################
database = "08_PreTrain"
collection = "train_data"

test_size = 0.25
random_state = 69

data = mc.getCollection(database, collection)

X = data[['onlyText']]
y = data[['identityMotive']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# we need to pass splitted sentences to the model
tokenized_sentences = X_train.apply(lambda row: word_tokenize(row['onlyText']), axis=1)

model = word2vec.Word2Vec(tokenized_sentences, min_count=5000, workers=12)

words = list(model.wv.vocab)
print(words)

# 2D PCA
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
