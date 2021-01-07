import logging
import os
import sys
from os.path import dirname
from underthesea.corpus.data import Sentence
from underthesea.models.text_classifier import TextClassifier
from underthesea.model_fetcher import ModelFetcher, UTSModel

FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('underthesea')

sys.path.insert(0, dirname(dirname(__file__)))
model_path = ModelFetcher.get_model_path(UTSModel.tc_bank)
classifier = None

sys.path.insert(0, dirname(dirname(__file__)))
classifier = None


def classify(X):
    global classifier
    model_name = 'TC_BANK_V131'
    model_path = ModelFetcher.get_model_path(model_name)

    if not classifier:
        if not os.path.exists(model_path):
            ModelFetcher.download(model_name)
        classifier = TextClassifier.load(model_path)

    sentence = Sentence(X)
    classifier.predict(sentence)
    labels = sentence.labels
    if not labels:
        return None
    labels = [label.value for label in labels]
    return labels
