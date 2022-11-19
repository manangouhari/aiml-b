import re
import nltk.data
import nltk


def clean_sentence(sentence):
    return re.sub(r'[\t\n\r=]', '', sentence)


def extract_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)


def extract_words(text):
    tokens = nltk.word_tokenize(text)
    return list(
        map(lambda x: clean_word(x),
            filter(lambda x: not x.startswith("'"), tokens)))


def contains_digit(word):

    for ch in word:
        if ch.isdigit(): return True
    return False


def clean_word(word):

    return re.sub(r'\'\"', '', word.lower())


def filter_words(words, stopwords):

    return list(
        filter(
            lambda x: x not in stopwords and not contains_digit(x) and '\''
            not in x, words))
