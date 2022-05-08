import keras

from default import *
from keras import Input, Model

from keras.layers import Conv1D
from keras.models import load_model
from keras.layers import Bidirectional

from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dense
from keras.layers.recurrent import LSTM


def train_cnnbilstm_model(n, weights, x_train, y_train):
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

    cnn = Conv1D(64, 3, padding='same', strides=1, activation='relu')(embed)

    bilstm = Bidirectional(LSTM(units=vocab_dim, dropout=0.5, activation='tanh', return_sequences=True))(cnn)

    flat = Flatten()(bilstm)

    tokens_output = Dense(4, activation='softmax')(flat)

    model = Model(inputs=tokens_input, outputs=tokens_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch)

    model.save('../models/cnnbilstm.h5')
