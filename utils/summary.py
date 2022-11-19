from collections import defaultdict
from math import log
from .cleanup import extract_words, filter_words


def get_tf_words(words, words_count, words_len):
    """
    Returns the tf value of all words.
    tf[word] = total appearances of the word / total words
    Parameters:
        words (list[str]): list of words for which term-frequency scores will be calculated.
        words_count ({str: int}): a frequency dictionary containing the data for how many times a word appears in the text.
        words_len (int): total number of words in the text.
    Returns:
        tf ({str: float}): dictionary containing tf scores for each word.
    """
    tf = defaultdict(float)
    for word in words_count:
        tf[word] = words_count[word] / words_len
    return tf


def get_tf_sentences(sentences, tf_words, stopwords):
    tf = defaultdict(float)
    for sentence in sentences:
        words_in_s = filter_words(extract_words(sentence), stopwords)
        tf[sentence] = sum(tf_words[word]
                           for word in words_in_s) / len(words_in_s)
    return tf


def get_idf_words(words, words_count, words_len, len_sentences):

    idf = defaultdict(float)
    for word in words_count:
        idf[word] = log(len_sentences / words_count[word], 10)
    return idf


def get_idf_sentences(sentences, idf_words, stopwords):

    idf = defaultdict(float)
    for sentence in sentences:
        words_in_s = filter_words(extract_words(sentence), stopwords)
        idf[sentence] = sum(idf_words[word]
                            for word in words_in_s) / len(words_in_s)
    return idf


def get_tfidf(sentences, sent_len, words, words_len, words_freq, stopwords):

    tf_words = get_tf_words(words, words_freq, words_len)
    tf_sentences = get_tf_sentences(sentences, tf_words, stopwords)

    idf_words = get_idf_words(words, words_freq, words_len, sent_len)
    idf_sentences = get_idf_sentences(sentences, idf_words, stopwords)

    tfidf = {s: (tf_sentences[s] * idf_sentences[s]) for s in sentences}

    return tfidf, tf_words