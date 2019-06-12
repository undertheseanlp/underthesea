from os.path import join, dirname

from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

from underthesea.model_fetcher import ModelFetcher, UTSModel

model_path = ModelFetcher.get_model_path(UTSModel.tc_svm_uts2017_bank_20190607)
classifer = TextClassifier.load(join(dirname(__file__), model_path))


def classify(X):
    sentence = Sentence(X)
    classifer.predict(sentence)
    labels = sentence.labels
    label_values = [label.value for label in labels]
    return label_values
