# https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/

## TODO: Balancing!

import scripts.mongoConnection as mc
import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math
import datetime
from sklearn.preprocessing import LabelEncoder
from keras.utils import plot_model
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from keras.preprocessing.sequence import pad_sequences

# GPU check
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

# Get data and prepare it
df = mc.getCollection("08_PreTrain", "train_data")
print("Data received")

df["onlyTextMotive"] = df["onlyText"] + df["identityMotive"]

df = df[['country', 'onlyTextMotive']]
df = df[df["country"] != 0]

print(df.shape)
print(df.columns.values)
print(df.country.unique())

# Train-test split
X = df["onlyTextMotive"]
y = df["country"]

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75, random_state=69)

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
    y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

# Encode labels
le = LabelEncoder()
# y_train_enc = pd.Series(le.fit_transform(y_train))
# y_test_enc = pd.Series(le.fit_transform(y_test))
y = pd.Series(le.fit_transform(y))
print("Data prepared")

print("Model (down)loading started...")
# Download BERT model
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=False)
print("BERT download done / BERT LOADED")

bert = BertTokenizer
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert(vocabulary_file, to_lower_case)

# Model parameters
BATCH_SIZE = 32
VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = len(df["country"].unique())
DROPOUT_RATE = 0.2
NB_EPOCHS = 5
print("Model parameters done")


# Tokenize comments
def tokenize_comments(comment):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment))


tokenized_comments = [tokenize_comments(comment) for comment in df["onlyTextMotive"]]

###########################
X_train_enc = [tokenize_comments(comment) for comment in X_train]
X_test_enc = [tokenize_comments(comment) for comment in X_test]
print("Comments tokenized")

# Make comment length equal
# comments_with_len_train = [[comment, y_train[i], len(comment)]
#                  for i, comment in enumerate(X_train_enc)]

# comments_with_len_test = [[comment, y_test[i], len(comment)]
#                  for i, comment in enumerate(X_test_enc)]
MAX_LEN = 512
input_ids_train = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in X_train_enc], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in X_test_enc], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in input_ids_train]
input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in input_ids_test]
input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# input_ids_train.sort(key=lambda x: x[2])
# input_ids_test.sort(key=lambda x: x[2])

sorted_comments_labels_train = [(comment_lab[0], comment_lab[1]) for comment_lab in input_ids_train]
sorted_comments_labels_test = [(comment_lab[0], comment_lab[1]) for comment_lab in input_ids_test]

processed_dataset_train = tf.data.Dataset.from_generator(lambda: sorted_comments_labels_train, output_types=(tf.int32, tf.int32))
processed_dataset_test = tf.data.Dataset.from_generator(lambda: sorted_comments_labels_test, output_types=(tf.int32, tf.int32))

batched_dataset_train = processed_dataset_train.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
batched_dataset_test = processed_dataset_test.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

##########################
# comments_with_len = [[comment, y[i], len(comment)] for i, comment in enumerate(tokenized_comments)]

# random.shuffle(comments_with_len)

# comments_with_len.sort(key=lambda x: x[2])

# sorted_comments_labels = [(comment_lab[0], comment_lab[1]) for comment_lab in comments_with_len]

# processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_comments_labels, output_types=(tf.int32, tf.int32))

# batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
print("Preprocessing done")

# Train test split
# TOTAL_BATCHES = math.ceil(len(sorted_comments_labels) / BATCH_SIZE)
# TEST_BATCHES = TOTAL_BATCHES // 25
# batched_dataset.shuffle(TOTAL_BATCHES)
# test_data = batched_dataset.take(TEST_BATCHES)
# train_data = batched_dataset.skip(TEST_BATCHES)
# print("Train test split done")


# Create model
class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=len(y.unique()),
                 dropout_rate=0.1,
                 training=True,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dimensions)
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=2, padding="valid", activation="relu")
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=3, padding="valid", activation="relu")
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=4, padding="valid", activation="relu")
        self.pool = tf.keras.layers.GlobalMaxPool1D()

        self.dense_1 = tf.keras.layers.Dense(units=dnn_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.last_dense = tf.keras.layers.Dense(units=model_output_classes, activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


print("Model function created")

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

# plot_model(text_model, to_file='model.png', show_shapes=True)
print("Model class instantiated")

text_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"])
print("Model compiled")

log_dir = "logs/bert/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print("Tensorboard logs created")

text_model.fit(processed_dataset_train, epochs=NB_EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard_callback])
print("Model fitted")

results = text_model.evaluate(processed_dataset_test)
print(results)
