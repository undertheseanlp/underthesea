import os
import sys
from os.path import dirname

from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

from underthesea.model_fetcher import UTSModel, ModelFetcher

sys.path.insert(0, dirname(__file__))

model_path = ModelFetcher.get_model_path(UTSModel.tc_svm_vntc_20190607)
if os.path.exists(model_path):
    classifier = TextClassifier.load(model_path)
else:
    print(f"{UTSModel.tc_svm_vntc_20190607.value} is not downloaded.\n"
          f"Download model with \"underthesea download {UTSModel.tc_svm_vntc_20190607.value}\"")


def classify(X):
    try:
        classifier
    except NameError:
        raise
    sentence = Sentence(X)
    classifier.predict(sentence)
    labels = sentence.labels
    return labels
