import csv

from nltk import word_tokenize
from nltk import FreqDist
import matplotlib.pyplot as plt


""" Input: Csv  Output: List """
def read_csv(file):
    sentences = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sentences.append(row['Chat'].lower())

    return sentences


""" Input: List  Output: List """
def tokenize_sentences(sentences):
    """

    :param sentences:
    :return:
    """
    words = []
    for sentence in sentences:
        cur_words = word_tokenize(sentence)
        words.extend(cur_words)

    return words


""" Input: List  Output: List """
def filter(words):
    # Remove punctuation
    punctuations = [',', '.', ':', ';', '?', '!']
    token1 = [word for word in words if word not in punctuations]

    # Remove stopwords
    # stopwords = [word.strip().lower() for word in open("stop.txt")]  # if there is txt file to save those stop words
    stopwords = ['and', 'the', 'a', 'to', 'is', 'ca', 'so', 'are',
                 'not', '\'m', 'can', 'n\'t', '\'s', 'be', 'through',
                 'it', 'should', 'with', 'am', 'of', 'about', 'but']
    token2 = [word for word in token1 if word not in stopwords]

    return token2


""" Input: List  Output: Dictionary """
def word_frequency_statistics(words):
    frequency = FreqDist(words)
    # print(frequency.items())

    return frequency


""" Input: Dictionary,int  Output: Image """
def draw_plot(frequency, n=20):
    fig = plt.figure()
    plt.gcf().subplots_adjust(bottom=0.165)  # avoids x-ticks cut-off
    frequency.plot(n, cumulative=False)
    plt.show()  # clear canvas at the same time
    fig.savefig('../images/frequency.png')

    return


""" Input: List  Output: Image """
"""
def generate_word_cloud(tokens):
    wordcloud = WordCloud(
        background_color="white",
        width=1500,
        height=960,
        margin=10
    ).generate(str(tokens))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    return"""


if __name__ == "__main__":
    s = read_csv('../corpus/data/chat.csv')
    w = tokenize_sentences(s)
    token = filter(w)
    f = word_frequency_statistics(token)
    draw_plot(f)
