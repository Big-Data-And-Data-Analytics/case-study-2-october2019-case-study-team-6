import csv
import tensorflow as tf
import numpy as np
from scripts.mongoConnection import getCollection, insertCollection
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional

vocab_size = 5000 # make the top list of words (common words)
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # OOV = Out of Vocabulary
training_portion = .9
df = getCollection('08_PreTrain', 'train_data')
posts = []
labels = []

for i in df.itertuples():
    labels.append(i[9])
    posts.append(i[5])

train_size = int(len(posts) * training_portion)
train_articles = posts[0: train_size]
train_labels = labels[0: train_size]
validation_articles = posts[train_size:]
validation_labels = labels[train_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

train_padded, x_test, training_label_seq, y_test = train_test_split(train_padded, training_label_seq, test_size = 0.1)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(7, activation='softmax'))
model.summary()

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, 
        validation_data=(validation_padded, validation_label_seq), verbose=2)

accr = model.evaluate(x_test, y_test)
print(f'loss: {accr[0]}, accuracy: {accr[1]}')