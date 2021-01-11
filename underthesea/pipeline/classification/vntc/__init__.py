import os
import sys
from os.path import dirname
from underthesea.corpus.data import Sentence
from underthesea.model_fetcher import ModelFetcher
from underthesea.models.text_classifier import TextClassifier

sys.path.insert(0, dirname(dirname(__file__)))
classifier = None


def classify(X):
    global classifier
    model_name = 'TC_GENERAL_V131'
    model_path = ModelFetcher.get_model_path(model_name)

    if not classifier:
        if not os.path.exists(model_path):
            ModelFetcher.download(model_name)
        classifier = TextClassifier.load(model_path)

    sentence = Sentence(X)
    classifier.predict(sentence)
    labels = sentence.labels
    return labels
