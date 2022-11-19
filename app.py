from flask import Flask, jsonify, request
from flask_cors import CORS

import spacy

import nltk
from nltk.corpus import stopwords
import nltk.data
from nltk.sentiment import SentimentIntensityAnalyzer

from collections import Counter

from utils.cleanup import clean_sentence, extract_sentences, extract_words, filter_words
from utils.sentiment import infer_sentiment
from utils.summary import get_tfidf

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_lg")


@app.before_first_request
def before_first_req():
    nltk.download("stopwords")
    nltk.download(["vader_lexicon"])


@app.route('/')
def index():
    return jsonify({'message': 'scrivi 2.0'})


@app.route('/analyse', methods=['POST'])
def analyse():
    try:
        data = request.json
        text = data["text"]

        STOPWORDS = stopwords.words('english')

        # Using Punkt. It is an unsupervised algorithm. We're using a pre-trained version of the model here.
        sentences = extract_sentences(text)

        # Using TreeBank Word Tokenizer & Punkt in combination. Treebank uses Regex.
        words = extract_words(text)
        words_freq = Counter(words)

        filtered = filter_words(words, STOPWORDS)
        tfidf, _ = get_tfidf(sentences, len(sentences), filtered,
                             len(filtered), words_freq, STOPWORDS)

        # Sentiment Analyser
        # VADER: Rule-based Model for Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        sentiment_dict = sia.polarity_scores(text)
        s = infer_sentiment(sentiment_dict['compound'])

        # INTENT
        # .similarity -> converts words to vectors and takes cosine similarity of the vectors to return a score.
        # word vectors -> multi-dimensional meaning representations of a word
        doc = nlp(text)
        # Decipher Intent
        intentAttrs = {
            "informative":
            doc.similarity(nlp("educate explain inform instruct")),
            "descriptive":
            doc.similarity(nlp("describe tell good bad new old")),
            "persuasive":
            doc.similarity(nlp("influence inspire entice convince"))
        }
        return jsonify({
            "stats": {
                "sentences": len(sentences),
                "words": len(words),
                "stopwords": len(words) - len(filtered),
                "freq": words_freq
            },
            'tfidf': {clean_sentence(k): v
                      for k, v in tfidf.items()},
            "sentiment": s,
            "intent": intentAttrs
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=8000, debug=True)