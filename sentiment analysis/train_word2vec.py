import numpy
import pandas
import keras
import os
import json

from default import *
from nltk import word_tokenize
from itertools import chain

from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


# prevent tf error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_file():
    """
    Loads csv file.
    :return:
    List, List
    """
    data = pandas.read_csv('../corpus/data/chat.csv', header=None, index_col=None)
    n, sentences, sources = len(data), data[0], data[2]  # number of chats, chats, source of sadness
    labels = numpy.zeros(n)

    for i in range(n):
        if sources[i].lower() == "school":
            labels[i] = 0
        elif sources[i].lower() == "relationship":
            labels[i] = 1
        elif sources[i].lower() == "value of life":
            labels[i] = 2
        else:
            labels[i] = 3

    return sentences[1:], labels[1:]  # remove header "Chat" and "Source"


def tokenizer(sentences):
    """
    Tokenizes the words from sentences.
    :param sentences: List
    :return:
    List
    """
    tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
    # tokens = list(chain.from_iterable(tokens))

    return tokens


def train_word2vec_model(data):
    """
    Trains word2vec models.
    :param data: List
    :return:

    """
    model = Word2Vec(vector_size=vocab_dim,
                     min_count=exposures,
                     window=local_window_size,
                     workers=cpu)
    model.build_vocab(data)
    # model.train(data, total_example=model.corpus_count, epochs=iterations)
    model.train(data, total_words=20, epochs=iterations)
    model.save('../models/word2vec.pkl')

    return model


def create_two_dictionaries(model=None, data=None):
    """
    Creates two dictionaries representing words in indexes and vectors respectively,
    and saves both as a txt files.
    :param model: word2vec
    :param data: tokens
    :return:
    Dictionary: from word to index
    Dictionary: from word to vecotr
    """
    if (data is not None) and (model is not None):
        gensim_dictionary = Dictionary()
        # fill the dictionary with bags of words method first, and allows it to be updated
        gensim_dictionary.doc2bow(model.wv.index_to_key, allow_update=True)  # only need key here
        # words with frequency lower than 10 -> 0, so there is k+1
        words_in_index = {v: k+1 for k,v in gensim_dictionary.items()}  # frequency higher than 10
        words_in_vector = {word: model.wv[word].tolist() for word in words_in_index.keys()}  # frequency higher than 10

        """
        # save dictionary(words->indexes) as a txt file
        d1 = open("../corpus/dictionary/words in index.txt", 'w', encoding='utf8')
        for word in words_in_index:
            d1.write(str(word))  # write words
            d1.write(' ')
            d1.write(str(words_in_index[word]))  # write index
            d1.write('\n')
        d1.close()

        # save dictionary(words->vectors) as a txt file
        d2 = open("../corpus/dictionary/words in vector.txt", 'w', encoding='utf8')
        for word in words_in_vector:
            d2.write(str(word))  # write words
            d2.write(' ')
            d2.write(str(words_in_vector[word]))  # write vector(embedding)
            d2.write('\n')
        d2.close()
        """

        return words_in_index, words_in_vector

    return


def write_index_dictionary(words_in_index):
    """
    Saves dictionary(words->indexes) as a json file.
    :param words_in_index:
    :return:
    """
    # dumps converts data to string
    # indent represents for the blank space at the header
    index_json = json.dumps(words_in_index, sort_keys=False, indent=4)
    file = open('../corpus/dictionary/words in index.json', 'w')
    file.write(index_json)
    file.close()

    return


def write_vector_dictionary(words_in_vector):
    """
    Saves dictionary(words->vectors) as a json file.
    :param words_in_vector:
    :return:
    """
    # dumps converts data to string
    # indent represents for the blank space at the header
    vector_json = json.dumps(words_in_vector, sort_keys=False, indent=4)
    file = open('../corpus/dictionary/words in vector.json', 'w')
    file.write(vector_json)
    file.close()

    return


def read_index_dictionary():
    """
    Reads word-index pairs from json file.
    :return:
    """
    file = open("../corpus/dictionary/words in index.json", encoding='utf8')
    dictionary = json.load(file)
    file.close()

    return dictionary


def read_vector_dictionary():
    """
    Reads word-vector pairs from json file.
    :return:
    """
    file = open("../corpus/dictionary/words in vector.json", encoding='utf8')
    dictionary = json.load(file)
    file.close()

    return {word: numpy.array(dictionary[word], dtype=float) for word in dictionary.keys()}


def parse_data(index_dictionary, data):
    """
    Covert words (in sentences) to indexes.
    :param index_dictionary:
    :param vector_dictionary:
    :param data: words
    :return:
    List
    """
    tokens_in_index = []
    for sentence in data:
        tokens = []
        for word in sentence:
            try:
                tokens.append(index_dictionary[word])  # word->index
            except:
                tokens.append(0)
        tokens_in_index.append(tokens)

    return sequence.pad_sequences(tokens_in_index, maxlen=max_length)  # padding in the same length


def process_data(index_dictionary, vector_dictionary, data, labels):
    """
    Splits dataset, uses one-hot encoding to represent labels.
    :param index_dictionary:
    :param vector_dictionary:
    :param data:
    :param labels:
    :return:
    int, list(V*N), list, list, list, list
    """
    n = len(index_dictionary)+1  # number of words (+1 for words of which frequency lower than 10)
    weights = numpy.zeros((n, vocab_dim))  # V*N
    for word, index in index_dictionary.items():
        weights[index, :] = vector_dictionary[word]

    # split dataset (train:test=8:2), x represents for the tokens, t represents for the labels
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train, num_classes=4)  # one-hot encoding
    y_test = keras.utils.to_categorical(y_test, num_classes=4)  # each vec in 4 dims

    return n, weights, x_train, y_train, x_test, y_test


def evaluate(model, x_test, y_test, score=True, report=False):
    """
    Loads model, predicts, calculated accuracy
    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    model = load_model(model)

    y_predict = model.predict(x_test)
    y_label = numpy.zeros(y_predict.shape)

    for i in range(len(y_predict)):
        max_value = max(y_predict[i])  # get maximum value as its final classification
        c = y_predict[i].tolist().index(max_value)
        y_label[i][c] = 1

    return accuracy_score(y_test, y_label) if score else None, classification_report(y_test, y_label) if report else None


def predict(model, tokens_in_index):
    """
    Predicts a single sentence.
    :param model:
    :param tokens_in_index:
    :return:
    """

    model = load_model(model)

    y_predict = model.predict(tokens_in_index)[0]  # [[0. 1. 0. 0.]] -> [0. 1. 0. 0.]
    print("S =", y_predict[0], " R =", y_predict[1], " V =", y_predict[2], " O =", y_predict[3])
    y_label = numpy.zeros(y_predict.shape)  # 4

    max_value = max(y_predict)  # get maximum value as its final classification
    c = y_predict.tolist().index(max_value)
    y_label[c] = 1

    if c == 0:
        source = "SCHOOL"
        response = "Take a deep breath, and relax. It isn't as bad as all that."
    elif c == 1:
        source = "RELATIONSHIP"
        response = "There's no need to worry. It'll turn out all right."
    elif c == 2:
        source = "VALUE OF LIFE"
        response = "I feel you. That must be so hard."
    else:
        source = "OTHERS"
        response = "I'm sorry about that. I know it sucks right now, but it'll get better with time."

    return source, response
