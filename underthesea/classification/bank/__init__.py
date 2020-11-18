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


def classify(X):
    global classifier

    if not classifier:
        if os.path.exists(model_path):
            classifier = TextClassifier.load(model_path)
        else:
            logger.error(
                f"Could not load model at {model_path}.\n"
                f"Download model with \"underthesea download {UTSModel.tc_bank.value}\".")
            sys.exit(1)
    sentence = Sentence(X)
    classifier.predict(sentence)
    labels = sentence.labels
    if not labels:
        return None
    return [label.value for label in labels]
