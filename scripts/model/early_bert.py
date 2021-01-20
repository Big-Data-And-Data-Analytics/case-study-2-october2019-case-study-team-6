import scripts.mongoConnection as mc
import tensorflow_hub as hub
#import bert
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import numpy as np
import random
import math


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
df = mc.getCollection("08_PreTrain", "train_data")

df["onlyTextMotive"] = df["onlyText"] + df["identityMotive"]

y = df['country']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
print("Data received")

bert = BertTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2", trainable=True)
print("BERT download done")

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert(vocabulary_file, to_lower_case)

# Tokenize comments
def tokenize_comments(comment):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comment))

tokenized_comments = [tokenize_comments(comment) for comment in df["onlyTextMotive"]]
print("Comments tokenized")

# Make comment length equal
comments_with_len = [[comment, y[i], len(comment)]
                 for i, comment in enumerate(tokenized_comments)]
print("Comment lenghts balanced")

random.shuffle(comments_with_len)

comments_with_len.sort(key=lambda x: x[2])

sorted_commentss_labels = [(comment_lab[0], comment_lab[1]) for comment_lab in comments_with_len]

processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_commentss_labels, output_types=(tf.int32, tf.int32))

BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
print("Preprocessing done")

# Train test split
TOTAL_BATCHES = math.ceil(len(sorted_commentss_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 25
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)
print("Train test split done")

#### CREATE THE MODEL ####
class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=True,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = tf.keras.layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = tf.keras.layers.GlobalMaxPool1D()
        
        self.dense_1 = tf.keras.layers.Dense(units=dnn_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        if model_output_classes == len(df["country"].unique()):
            self.last_dense = tf.keras.layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = tf.keras.layers.Dense(units=model_output_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output
print("Model function created")

# Architecture of BERT
# https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/

VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = len(df["country"].unique())

DROPOUT_RATE = 0.2

NB_EPOCHS = 5
print("Model parameters")

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)
print("Model class instaciated")

if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])

text_model.fit(train_data, epochs=NB_EPOCHS)
print("Model fitted")

results = text_model.evaluate(test_data)
print(results)
