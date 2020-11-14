import os
import sys
from os.path import dirname
import logging
from underthesea.corpus.data import Sentence
from underthesea.model_fetcher import UTSModel, ModelFetcher
from underthesea.models.text_classifier import TextClassifier

FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('underthesea')

sys.path.insert(0, dirname(dirname(__file__)))

model_path = ModelFetcher.get_model_path(UTSModel.tc_general)

classifier = None


def classify(X):
    global classifier

    if not classifier:
        if os.path.exists(model_path):
            classifier = TextClassifier.load(model_path)
        else:
            logger.error(
                f"Could not load model at {model_path}.\n"
                f"Download model with \"underthesea download {UTSModel.tc_general.value}\".")
            sys.exit(1)

    sentence = Sentence(X)
    classifier.predict(sentence)
    labels = sentence.labels
    return labels
