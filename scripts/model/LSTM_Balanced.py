import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import NearMiss, TomekLinks
from scripts.mongoConnection import getCollection
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (LSTM, Activation, Bidirectional, Dense,
                                     Dropout, Embedding, Flatten)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

filepath_lstm_model = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/lstm_balanced_model"
vocab_size = 5000 # make the top list of words (common words)
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # OOV = Out of Vocabulary
training_portion = .9
df = getCollection('08_PreTrain', 'train_data')

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['onlyText'].values)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(df['onlyText'].values)
X = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_token = Tokenizer()
label_token.fit_on_texts(df['identityMotive'].values)

Y = np.array(label_token.texts_to_sequences(df['identityMotive'].values))

X, x_test, Y, y_test = train_test_split(X, Y, test_size = 0.2)
print(f'{X.shape} shape of x; {Y.shape} is shape of y')

tm = SMOTEENN()
print("TM started")
X_sm, y_sm = tm.fit_resample(X, Y)
print(f'{X_sm.shape} shape of x; {y_sm.shape} is shape of y')

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

num_epochs = 5
history = model.fit(X_sm, y_sm, epochs=num_epochs, validation_split = 0.1, verbose=2)

accr = model.evaluate(x_test, y_test)
print(f'loss: {accr[0]}, accuracy: {accr[1]}')

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plot_graphs(history, 'accuracy')
plt.subplot(1,2,2)
plot_graphs(history, 'loss')

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# Save model
model.save(filepath_lstm_model)

# Load model
model = tf.keras.models.load_model(filepath_lstm_model)

# ## Prediction and some evaluation
# pred_input = tm.fit("All teams did a great job, thats nice football to watch", "meaning")

# predictions = model.predict(np.array([pred_input]))
# print(predictions)
