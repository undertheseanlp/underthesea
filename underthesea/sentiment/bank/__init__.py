from os.path import join, dirname
import sys
from languageflow.data import Sentence
from languageflow.models.text_classifier import TextClassifier

sys.path.insert(0, dirname(__file__))

model_folder = 'sentiment_svm_uts2017_bank_sa'
classifier = TextClassifier.load(join(dirname(__file__), model_folder))


def sentiment(text):
    print(f"\nText: {text}")
    sentence = Sentence(text)
    classifier.predict(sentence)
    labels = sentence.labels
    print(f"Labels: {labels[0]}")
