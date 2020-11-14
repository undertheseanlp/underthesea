import logging
import os
import sys
from underthesea.corpus.data import Sentence
from underthesea.model_fetcher import ModelFetcher, UTSModel
from underthesea.models.text_classifier import TextClassifier
from . import text_features

FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('underthesea')

sys.modules['text_features'] = text_features
model_path = ModelFetcher.get_model_path(UTSModel.sa_bank)
classifier = None


def sentiment(text):
    global classifier

    if not classifier:
        if os.path.exists(model_path):
            classifier = TextClassifier.load(model_path)
        else:
            logger.error(
                f"Could not load model at {model_path}.\n"
                f"Download model with \"underthesea download {UTSModel.sa_bank.value}\".")
            sys.exit(1)
    sentence = Sentence(text)
    classifier.predict(sentence)
    labels = sentence.labels
    if labels is None:
        return None
    return [label.value for label in labels]
