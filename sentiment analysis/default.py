import multiprocessing

# pre-defined
cpu = multiprocessing.cpu_count()  # threads
vocab_dim = 50  # dimension of the output word vector, number of the units in hidden layer
exposures = 2  # words of which frequency is lower than 2 will be ignored
local_window_size = 6
iterations = 1

max_length = 20  # max length of an input sentence
batch_size = 4
n_epoch = 30
