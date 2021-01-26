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
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class NN_Model:
    def __init__(self, tensorboard, vocab_size=5000, embedding_dim=64, max_length=200, trunc_type='post',
                 padding_type='post', oov_tok='<OOV>', training_portion=.9, num_epochs=5, dropout=0.0):
        self.vocab_size = vocab_size  # make the top list of words (common words)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.trunc_type = trunc_type
        self.padding_type = padding_type
        self.oov_tok = oov_tok  # OOV = Out of Vocabulary
        self.training_portion = training_portion
        self.num_epochs = num_epochs
        self.tensorboard = tensorboard
        self.dropout = dropout

    def tokenize(self, df):
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok)
        tokenizer.fit_on_texts(df['onlyText'].values)
        # word_index = tokenizer.word_index

        train_sequences = tokenizer.texts_to_sequences(df['onlyText'].values)
        self.X = pad_sequences(train_sequences, maxlen=self.max_length, padding=self.padding_type,
                               truncating=self.trunc_type)

        label_token = Tokenizer()
        label_token.fit_on_texts(df['identityMotive'].values)
        self.Y = np.array(label_token.texts_to_sequences(df['identityMotive'].values))

    def lstm(self):
        X, x_test, Y, y_test = train_test_split(self.X, self.Y, test_size=0.2)
        print(f'{X.shape} shape of x; {Y.shape} is shape of y')

        # tm = balancing()
        # X_sm, y_sm = tm.fit_resample(X, Y)
        # print(f'{X_sm.shape} shape of x; {y_sm.shape} is shape of y')

        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_dim))
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(self.embedding_dim)))
        model.add(Dense(7, activation='softmax'))
        model.summary()

        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'],
        )

        # history = model.fit(X_sm, y_sm, epochs=self.num_epochs, validation_split=0.1, verbose=2)
        history = model.fit(X, Y, epochs=self.num_epochs, validation_split=0.1, verbose=2, callbacks=[self.tensorboard])
        accr = model.evaluate(x_test, y_test)
        print(f'loss: {accr[0]}, accuracy: {accr[1]}')


if __name__ == '__main__':
    df = getCollection('08_PreTrain', 'train_data')
    em = [32, 64, 128, 256]
    ml = [32, 64, 128, 256]
    tp = [.6, 0.75, .9]
    vs = [500, 1000, 2000, 4000, 8000, 10000]

    counter = 0

    for vc_size in vs:
        for emb in em:
            for max_l in ml:
                for trp in tp:
                    print(vc_size, emb, max_l, trp)
                    print(f"Model Number: {counter}")
                    counter += 1
                    NAME = f"Model_Vocabsize_{vc_size}_Embdim_{emb}_maxlen_{max_l}_trp_{trp}"

                    tb = TensorBoard(log_dir="logs_lstm/{}".format(NAME))
                    nn = NN_Model(vocab_size=vc_size, embedding_dim=emb, max_length=max_l, trunc_type='post',
                        padding_type='post', oov_tok='<OOV>', training_portion=tp, num_epochs=5, tensorboard=tb)
                    nn.tokenize(df)
                    nn.lstm()
    # nn = NN_Model(vocab_size=5000, embedding_dim=64, max_length=200, training_portion=.9, num_epochs=5,
    # tensorboard=tb)
    # nn.tokenize(df)
    # nn.lstm()
