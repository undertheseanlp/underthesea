from os.path import join, dirname
import sys

from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

sys.path.insert(0, dirname(__file__))


def classify(X):
    classifer = TextClassifier.load(join(dirname(__file__), 'classification_svm_uts2017_bank'))
    sentence = Sentence(X)
    classifer.predict(sentence)
    label = sentence.labels
    return label
