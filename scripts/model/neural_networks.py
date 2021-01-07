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

class NN_Model:
    def __init__(self) -> None:
        self.vocab_size = 5000 # make the top list of words (common words)
        self.embedding_dim = 64
        self.max_length = 200
        self.trunc_type = 'post'
        self.padding_type = 'post'
        self.oov_tok = '<OOV>' # OOV = Out of Vocabulary
        self.training_portion = .9
        self.num_epochs = 5
    

    def tokenize(self, df):
        tokenizer = Tokenizer(num_words = self.vocab_size, oov_token=self.oov_tok)
        tokenizer.fit_on_texts(df['onlyText'].values)
        word_index = tokenizer.word_index

        train_sequences = tokenizer.texts_to_sequences(df['onlyText'].values)
        self.X = pad_sequences(train_sequences, maxlen=self.max_length, padding=self.padding_type, truncating=self.trunc_type)

        label_token = Tokenizer()
        label_token.fit_on_texts(df['identityMotive'].values)

        self.Y = np.array(label_token.texts_to_sequences(df['identityMotive'].values))


    def lstm(self, balancing):
        X, x_test, Y, y_test = train_test_split(self.X, self.Y, test_size = 0.2)
        print(f'{X.shape} shape of x; {Y.shape} is shape of y')

        tm = balancing()
        X_sm, y_sm = tm.fit_resample(X, Y)
        print(f'{X_sm.shape} shape of x; {y_sm.shape} is shape of y')

        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_dim))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(self.embedding_dim)))
        model.add(Dense(7, activation='softmax'))
        model.summary()

        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'],
        )

        history = model.fit(X_sm, y_sm, epochs=self.num_epochs, validation_split = 0.1, verbose=2)
        accr = model.evaluate(x_test, y_test)
        print(f'loss: {accr[0]}, accuracy: {accr[1]}')

if __name__ == '__main__':
    df = getCollection('08_PreTrain', 'train_data')
    nn = NN_Model()
    nn.tokenize(df)
    nn.lstm(SMOTEENN)
    