# https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/

## TODO: Balancing!

import scripts.mongoConnection as mc
import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf
import random
import math
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(log_dir="logs_BERT/{}".format("BERT_model"))
filepath_bert_model = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/bert_model"

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

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["country"]) # Use this for later predicting
# tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32') # Can also use this instead of above line
print("Data prepared")

# Download BERT model
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=True)
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

print("Comments tokenized")

# Make comment length equal
comments_with_len = [[comment, y[i], len(comment)]
                 for i, comment in enumerate(tokenized_comments)]

print("Comment lenghts balanced")

random.shuffle(comments_with_len)
comments_with_len.sort(key=lambda x: x[2])
sorted_commentss_labels = [(comment_lab[0], comment_lab[1]) for comment_lab in comments_with_len]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_commentss_labels, output_types=(tf.int32, tf.int32))
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

print("Preprocessing done")

# Train test split
TOTAL_BATCHES = math.ceil(len(sorted_commentss_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 25
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

print("Train test split done")

# Create model
class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=len(df["country"].unique()),
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
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
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

print("Model class instanciated")

text_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"])
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

text_model.fit(train_data, epochs=NB_EPOCHS, batch_size=BATCH_SIZE, callbacks=[tb])
print("Model fitted")

results = text_model.evaluate(test_data)
print(results)

# Save model
text_model.save(filepath=filepath_bert_model)
print("Model saved")

# Load model
model = tf.keras.models.load_model(filepath=filepath_bert_model)
print("Model loaded")

# Predict
new_input = "I love the german football team"
tokenized_new_input = [tokenize_comments(comment) for comment in new_input]

new_input_same_length = [[comment, y[i], len(comment)]
                 for i, comment in enumerate(tokenized_new_input)]

random.shuffle(new_input_same_length)
new_input_same_length.sort(key=lambda x: x[2])
sorted_commentss_labels = [(comment_lab[0], comment_lab[1]) for comment_lab in new_input_same_length]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_commentss_labels, output_types=(tf.int32, tf.int32))
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

model.predict(np.array([new_input]))
