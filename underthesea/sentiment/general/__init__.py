import logging
import os
from os.path import dirname
import sys
from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier
from underthesea.model_fetcher import ModelFetcher, UTSModel


FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('underthesea')

sys.path.insert(0, (dirname(__file__)))
model_path = ModelFetcher.get_model_path(UTSModel.sa_general)
classifier = None


def sentiment(text):
    global classifier

    if not classifier:
        if os.path.exists(model_path):
            classifier = TextClassifier.load(model_path)
        else:
            logger.error(
                f"Could not load model at {model_path}.\n"
                f"Download model with \"underthesea download {UTSModel.sa_general.value}\".")
            sys.exit(1)
    sentence = Sentence(text)
    classifier.predict(sentence)
    label = sentence.labels[0]
    if label == "1":
        label = "negative"
    if label == "0":
        label = "positive"
    return label
