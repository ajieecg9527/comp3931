import time  # record

from train_word2vec import *
from train_textcnn import train_textcnn_model
from train_cnnbilstm import train_cnnbilstm_model
from numpy import mean



def train_word2vec():
    print("Training Word2Vec Model...")

    print("Loading corpus...")
    sentences, labels = load_file()
    # print(sentences)
    # print(labels)

    print("Preprocessing corpus...")
    tokens = tokenizer(sentences)

    print("Training model...")
    model = train_word2vec_model(tokens)

    print("Creating two dictionaries...")
    words_in_index, words_in_vector = create_two_dictionaries(model, tokens)

    print("Saving two dictionaries as json files...\n")
    write_index_dictionary(words_in_index)
    write_vector_dictionary(words_in_vector)

    return tokens


def train_textcnn():
    print("Training Text CNN Model...")

    print("Loading corpus...")
    sentences, labels = load_file()

    print("Preprocessing corpus...")
    tokens = tokenizer(sentences)

    print("Loading dictionaries...")
    index_dict, vector_dict = read_index_dictionary(), read_vector_dictionary()

    print("Converting word to index...")
    tokens_in_index = parse_data(index_dict, tokens)

    print("Spliting dataset into training set and testing set...")
    V, embedding_weights, x_train, y_train, x_test, y_test = process_data(index_dict, vector_dict, tokens_in_index,
                                                                          labels)

    print("Training model...")
    train_textcnn_model(V, embedding_weights, x_train, y_train)

    print("Evaluating model accuracy...")
    accuracy, report = evaluate('../models/textcnn.h5', x_test, y_test, report=True)
    print("Accuracy:", accuracy)
    print("Reprot:", report, "\n")


def evaluate_textcnn(n):
    """
    Records time of training textcnn model for n iterations,
    and computes an average of accuracy.
    :return:
    """
    print("Recording Time of Training Text CNN Model...")

    # pre-process
    sentences, labels = load_file()
    tokens = tokenizer(sentences)
    index_dict, vector_dict = read_index_dictionary(), read_vector_dictionary()
    tokens_in_index = parse_data(index_dict, tokens)
    V, embedding_weights, x_train, y_train, x_test, y_test = process_data(index_dict, vector_dict, tokens_in_index,
                                                                          labels)
    time_list = []
    accuracy_list = []

    for i in range(n):
        print("Iteration: ", i)

        # the exact training time
        start_time = time.time()
        train_textcnn_model(V, embedding_weights, x_train, y_train)
        end_time = time.time()

        accuracy, _ = evaluate('../models/textcnn.h5', x_test, y_test)
        print("Accuracy:", accuracy, "\n")

        time_list.append(end_time-start_time)
        accuracy_list.append(accuracy)

    return mean(time_list), mean(accuracy_list)


def train_cnnbilstm():
    print("Training CNN Bi-LSTM Model...")

    print("Loading corpus...")
    sentences, labels = load_file()

    print("Preprocessing corpus...")
    tokens = tokenizer(sentences)

    print("Loading dictionaries...")
    index_dict, vector_dict = read_index_dictionary(), read_vector_dictionary()

    print("Converting word to index...")
    tokens_in_index = parse_data(index_dict, tokens)

    print("Spliting dataset into training set and testing set...")
    V, embedding_weights, x_train, y_train, x_test, y_test = process_data(index_dict, vector_dict, tokens_in_index,
                                                                          labels)

    print("Training model...")
    train_cnnbilstm_model(V, embedding_weights, x_train, y_train)

    print("Evaluating model accuracy...")
    accuracy, _ = evaluate('../models/cnnbilstm.h5', x_test, y_test)
    print("Accuracy:", accuracy)


def evaluate_cnnlstm(n):
    """
    Records time of training cnnlstm model for n iterations,
    and computes an average of accuracy.
    :return:
    """
    print("Recording Time of Training CNN LSTM Model...")

    # pre-process
    sentences, labels = load_file()
    tokens = tokenizer(sentences)
    index_dict, vector_dict = read_index_dictionary(), read_vector_dictionary()
    tokens_in_index = parse_data(index_dict, tokens)
    V, embedding_weights, x_train, y_train, x_test, y_test = process_data(index_dict, vector_dict, tokens_in_index,
                                                                          labels)
    time_list = []
    accuracy_list = []

    for i in range(n):
        print("Iteration: ", i)

        # the exact training time
        start_time = time.time()
        train_cnnbilstm_model(V, embedding_weights, x_train, y_train)
        end_time = time.time()

        accuracy, _ = evaluate('../models/cnnbilstm.h5', x_test, y_test)
        print("Accuracy:", accuracy, "\n")

        time_list.append(end_time-start_time)
        accuracy_list.append(accuracy)

    return mean(time_list), mean(accuracy_list)


def apply_textcnn(sentence):
    # print("Applying Text CNN Model...")

    # print("Preprocessing the input sentence...")
    tokens = tokenizer([sentence])

    # print("Loading dictionaries...")
    index_dict, vector_dict = read_index_dictionary(), read_vector_dictionary()

    # print("Converting word to index...")
    tokens_in_index = parse_data(index_dict, tokens)

    # print("Identifying source...")
    return predict('../models/textcnn.h5', tokens_in_index)


def apply_cnnbitlstm(sentence):
    # print("Applying CNN Bi-LSTM Model...")

    # print("Preprocessing the input sentence...")
    tokens = tokenizer([sentence])

    # print("Loading dictionaries...")
    index_dict, vector_dict = read_index_dictionary(), read_vector_dictionary()

    # print("Converting word to index...")
    tokens_in_index = parse_data(index_dict, tokens)

    # print("Identifying source...")
    return predict('../models/cnnbilstm.h5', tokens_in_index)


def chat():
    """
    Initiate a chat.
    :return:
    """
    print("Initiate a chat with AI Ayato.")
    print("Ayato: Greetings! I'm here to listen to your trouble.")
    for i in range(3):
        chat = input("You: ")
        if chat.lower() == "exit":
            break
        else:
            q, a = apply_textcnn(chat)
            print("[ Ayato identifies the source of your trouble is", q, "]")
            print("Ayato:", a)
    print("Thanks for using AI Ayato :D")


if __name__ == "__main__":
    # train_word2vec()
    # train_textcnn()  # excellent for short text classification (real-time chat)
    # train_cnnbilstm()  # run with longer training time but almost same accuracy
    chat()

    # time_textcnn, accuracy_textcnn = evaluate_textcnn(10)
    # print("Time: ", time_textcnn, "\nAccuracy", accuracy_textcnn)

    # time_cnnlstm, accuracy_cnnlstm = evaluate_cnnlstm(10)
    # print("Time: ", time_cnnlstm, "\nAccuracy", accuracy_cnnlstm)

