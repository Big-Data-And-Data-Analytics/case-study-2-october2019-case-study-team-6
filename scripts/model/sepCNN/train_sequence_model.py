"""Module to train sequence model.

Vectorizes training and validation texts into sequences and uses that for
training a sequence model - a sepCNN model. We use sequence model for text
classification when the ratio of number of samples to number of words per
sample for the given dataset is very large (>~15K).
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from scripts.mongoConnection import getCollection
import tensorflow as tf
from scripts.model.sepCNN import build_model
from scripts.model.sepCNN import load_data
from scripts.model.sepCNN import vectorize_data
from scripts.model.sepCNN import explore_data
from tensorflow.keras.callbacks import TensorBoard

FLAGS = None

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000


def train_sequence_model(data,
                         learning_rate=1e-3,
                         # learning_rate=0.001,
                         epochs=10,
                         batch_size=128,
                         blocks=2,
                         filters=64,
                         dropout_rate=0.2,
                         embedding_dim=2000,
                         kernel_size=3,
                         pool_size=3, tensorboard=None):
    """Trains sequence model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
            unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val, word_index = vectorize_data.sequence_vectorize(
        train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    # Create model instance.
    model = build_model.sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    # callbacks = [tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=2), tensorboard]

    callbacks = [tensorboard]

    # Train and validate model.
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=1,  # Logs once per epoch.
        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    # model.save('rotten_tomatoes_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='./data',
    #                     help='input data directory')
    # FLAGS, unparsed = parser.parse_known_args()
    # data_dir = "C:/Users/shubham/Downloads/sentiment-analysis-on-movie-reviews/"

    # Using the Rotten tomatoes movie reviews dataset to demonstrate
    # training sequence model.
    test_size = 0.25
    random_state = 69
    data = getCollection(db="08_PreTrain", col="train_data")
    data = data[data["country"] != "NORTHEN IRELAND"]
    data = data[data["country"] != 0]
    le = LabelEncoder()
    data['country'] = le.fit_transform(data['country'])

    # data = load_data.load_rotten_tomatoes_sentiment_analysis_dataset(data_dir)
    # print(data)

    # data = data[data["country"] != 0]

    # TODO Add funtionality that it will automatically filter out countries below threshhold of 4

    X = data['onlyText'] + " " + data['identityMotive']
    y = data['country']
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, train_size=1 - test_size,
                                 random_state=random_state)

    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
        y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]

    data = (X_train, y_train), (X_test, y_test)

    # FLAGS.data_dir)
    # NAME = "testLog"
    # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    # train_sequence_model(data, tensorboard=tensorboard)

    embedding_dim = [200, 2000, 4000, 8000]
    batch_size = [32, 64, 128]
    dropout_rate = [0.2, 0.5]
    learning_rate = [1e-3, 1e-2, 1e-1]
    blocks = 3
    pool_size = [3]
    epochs = 50
    filters = 7
    kernel_size = 3

    for em in embedding_dim:
        for dr in dropout_rate:
            for bs in batch_size:
                for lr in learning_rate:
                    for pool in pool_size:
                        print(em, dr, bs, lr, pool, epochs, filters, kernel_size)
                        NAME = f"sepCNN_lr{lr}_em{em}_dr{dr}_bs_{bs}_pool{pool}_{epochs}_{filters}_{kernel_size}"
                        tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
                        train_sequence_model(data,
                                                 learning_rate=lr,
                                                 # learning_rate=0.001,
                                                 epochs=10,
                                                 batch_size=bs,
                                                 blocks=3,
                                                 filters=7,
                                                 dropout_rate=dr,
                                                 embedding_dim=em,
                                                 kernel_size=3,
                                                 pool_size=pool, tensorboard=tensorboard)
