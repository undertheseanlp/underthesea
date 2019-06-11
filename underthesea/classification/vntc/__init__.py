from os.path import join, dirname
import sys

from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

from underthesea.model_fetcher import UTSModel, ModelFetcher

sys.path.insert(0, dirname(__file__))

model_path = ModelFetcher.load_model(UTSModel.tc_svm_vntc_20190607)
classifer = TextClassifier.load(join(dirname(__file__), model_path))


def classify(X):
    sentence = Sentence(X)
    classifer.predict(sentence)
    labels = sentence.labels
    return labels
