from os.path import join, dirname
import sys
from languageflow.data import Sentence
from underthesea.model_fetcher import ModelFetcher, UTSModel


sys.path.insert(0, dirname(__file__))

model_folder = 'se_svm_2019-06'
classifier = ModelFetcher.load_model(UTSModel.se_svm_bank_2019_06)


def sentiment(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    labels = sentence.labels
    return labels
