import keras

from default import *
from keras import Input, Model

from keras.layers import Conv1D, MaxPooling1D, concatenate
from keras.models import load_model

from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dropout, Dense


def train_textcnn_model(n, weights, x_train, y_train):
    """
    Reproduces Text CNN model.
    Text CNN model deals with short-text classification.
    :param index_dictionary:
    :param vector_dictionary:
    :param data:
    :param labels:
    :return:
    """

    tokens_input = Input(shape=(max_length,), dtype='float64')  # 20 dims
    embedder = Embedding(input_dim=n,  # numbers of words in dictionary
                         input_length=max_length,  # fixed length in input
                         weights=[weights],
                         output_dim=vocab_dim)  # hidden layer units

    embed = embedder(tokens_input)

    cnn1 = Conv1D(64, 3, padding='same', strides=1, activation='relu')(embed)
    cnn2 = Conv1D(64, 4, padding='same', strides=1, activation='relu')(embed)
    cnn3 = Conv1D(64, 5, padding='same', strides=1, activation='relu')(embed)

    cnn1 = MaxPooling1D(pool_size=9)(cnn1)
    cnn2 = MaxPooling1D(pool_size=8)(cnn2)
    cnn3 = MaxPooling1D(pool_size=7)(cnn3)

    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

    flat = Flatten()(cnn)
    drop = Dropout(0.5)(flat)
    tokens_output = Dense(4, activation='softmax')(drop)

    model = Model(inputs=tokens_input, outputs=tokens_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch)

    model.save('../models/textcnn.h5')
