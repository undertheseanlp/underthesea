import os
import sys
from underthesea.corpus.data import Sentence
from underthesea.model_fetcher import ModelFetcher
from underthesea.models.text_classifier import TextClassifier
from os.path import dirname

sys.path.insert(0, dirname(dirname(__file__)))
classifier = None


def sentiment(X):
    global classifier
    model_name = 'SA_GENERAL_V131'
    model_path = ModelFetcher.get_model_path(model_name)

    if not classifier:
        if not os.path.exists(model_path):
            ModelFetcher.download(model_name)
        classifier = TextClassifier.load(model_path)

    sentence = Sentence(X)
    classifier.predict(sentence)
    labels = sentence.labels
    try:
        label_map = {'POS': 'positive', 'NEG': 'negative'}
        label = label_map[labels[0]]
        return label
    except Exception:
        return None
