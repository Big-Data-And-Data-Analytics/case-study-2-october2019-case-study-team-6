import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from scripts.mongoConnection import getCollection
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras import preprocessing as kprocessing
from tensorflow.keras.layers import (GRU, LSTM, Activation, BatchNormalization,
                                     Bidirectional, Conv1D, Dense, Dropout,
                                     Embedding, Flatten,
                                     GlobalAveragePooling1D, Input, MaxPool1D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

df = getCollection('08_PreTrain', 'train_data')
nlp = KeyedVectors.load_word2vec_format('D:/case study/case-study-2-october2019-case-study-team-6/scripts/model/GoogleNews-vectors-negative300.bin.gz', binary=True, encoding='utf-8')
df_train, df_test = train_test_split(df, test_size=0.1)
y_train = df_train['identityMotive'].values
y_test = df_test['identityMotive'].values
corpus = df_train['onlyText']
lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

tokenizer = Tokenizer(lower=True, split=' ', 
                     oov_token="NaN", 
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
X_train = pad_sequences(lst_text2seq, maxlen=30, padding="post", truncating="post")
corpus = df_test['onlyText']
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, 
                 len(lst_words), 1)]
    lst_corpus.append(lst_grams)

lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)
X_test = pad_sequences(lst_text2seq, maxlen=30,
             padding="post", truncating="post")
embeddings = np.zeros((len(dic_vocabulary)+1, 300))
for word,idx in dic_vocabulary.items():
    ## update the row with vector
    try:
        embeddings[idx] =  nlp[word]
    ## if word not in model then skip and the row stays all 0s
    except:
        pass

x_in = layers.Input(shape=(30,))
x = layers.Embedding(input_dim=embeddings.shape[0],  
                     output_dim=embeddings.shape[1], 
                     weights=[embeddings],
                     input_length=30, trainable=False)(x_in)
x = layers.Bidirectional(layers.LSTM(units=50, dropout=0.2, 
                         return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=50, dropout=0.2))(x)
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(7, activation='softmax')(x)
opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)
model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

model.summary()
dic_y_mapping = {n:label for n,label in 
                 enumerate(np.unique(y_train))}
inverse_dic = {v:k for k,v in dic_y_mapping.items()}
y_train = np.array([inverse_dic[y] for y in y_train])

dic_y_test_mapping = {n:label for n,label in 
                 enumerate(np.unique(y_test))}
inverse_dic = {v:k for k,v in dic_y_test_mapping.items()}
y_test = np.array([inverse_dic[y] for y in y_test])

training = model.fit(x=X_train, y=y_train, batch_size=256, 
                     epochs=10, shuffle=True, verbose=0, 
                     validation_split=0.1)
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
ax11.set_ylabel("Score", color='steelblue')
ax11.legend()
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics:
     ax22.plot(training.history['val_'+metric], label=metric)
ax22.set_ylabel("Score", color="steelblue")
plt.show()

predicted_prob = model.predict(X_test)
predicted = [dic_y_mapping[np.argmax(pred)] for pred in 
             predicted_prob]
accuracy = accuracy_score(y_test, predicted)
print(accuracy)