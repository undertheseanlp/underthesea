from enum import Enum
from os.path import join
import json
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

from underthesea.corpus.data import Sentence, Label

warnings.filterwarnings("ignore", ".*", UserWarning)


class Model:
    pass


class TEXT_CLASSIFIER_ESTIMATOR(Enum):
    FAST_TEXT = "FAST_TEXT"
    SVC = "SVC"
    PIPELINE = "PIPELINE"


class TextClassifier(Model):

    def __init__(self, estimator: TEXT_CLASSIFIER_ESTIMATOR, multilabel=False, **params):
        self.multilabel = multilabel
        if multilabel:
            self.y_encoder = MultiLabelBinarizer()
        self.estimator = estimator
        self.params = params
        if estimator == TEXT_CLASSIFIER_ESTIMATOR.FAST_TEXT:
            self.ft = None
        if estimator == TEXT_CLASSIFIER_ESTIMATOR.SVC:
            self.svc = None
        if estimator == TEXT_CLASSIFIER_ESTIMATOR.PIPELINE:
            if "pipeline" in params:
                self.pipeline = params["pipeline"]
            else:
                self.pipeline = None

    @staticmethod
    def load(model_folder):
        with open(join(model_folder, "metadata.json")) as f:
            metadata = json.loads(f.read())
        if metadata['estimator'] == 'SVC':
            estimator = TEXT_CLASSIFIER_ESTIMATOR.SVC
        if metadata['estimator'] == 'FAST_TEXT':
            estimator = TEXT_CLASSIFIER_ESTIMATOR.FAST_TEXT
        if metadata['estimator'] == 'PIPELINE':
            estimator = TEXT_CLASSIFIER_ESTIMATOR.PIPELINE

        # GH-304: remove fasttext
        # if estimator == TEXT_CLASSIFIER_ESTIMATOR.FAST_TEXT:
        #     model_file = join(model_folder, "model.bin")
        #     classifier = TextClassifier(estimator=TEXT_CLASSIFIER_ESTIMATOR.FAST_TEXT)
        #     classifier.ft = fastText.load_model(model_file)
        #     return classifier

        if estimator == TEXT_CLASSIFIER_ESTIMATOR.SVC:
            classifier = TextClassifier(estimator=TEXT_CLASSIFIER_ESTIMATOR.SVC)
            classifier.svc = joblib.load(join(model_folder, "estimator.joblib"))
            x_transformer = joblib.load(join(model_folder, "x_transformer.joblib"))
            classifier.x_transformer = x_transformer
            y_transformer = joblib.load(join(model_folder, "y_transformer.joblib"))
            classifier.y_transformer = y_transformer
            return classifier

        if estimator == TEXT_CLASSIFIER_ESTIMATOR.PIPELINE:
            classifier = TextClassifier(estimator=TEXT_CLASSIFIER_ESTIMATOR.PIPELINE)
            if "multilabel" in metadata:
                if metadata["multilabel"]:
                    classifier.multilabel = True
                    classifier.y_encoder = joblib.load(join(model_folder, "y_encoder.joblib"))
            classifier.pipeline = joblib.load(join(model_folder, "pipeline.joblib"))

            return classifier

    def predict(self, sentence: Sentence):
        if self.estimator == TEXT_CLASSIFIER_ESTIMATOR.FAST_TEXT:
            values, scores = self.ft.predict(sentence.text)
            labels = []
            for value, score in zip(values, scores):
                value = value.replace("__label__", "")
                label = Label(value, score)
                labels.append(label)
            sentence.add_labels(labels)

        if self.estimator == TEXT_CLASSIFIER_ESTIMATOR.SVC:
            text = sentence.text
            X = self.x_transformer.transform([text])
            y = self.svc.predict(X)
            y = self.y_transformer.inverse_transform(y)
            sentence.add_labels(y)

        if self.estimator == TEXT_CLASSIFIER_ESTIMATOR.PIPELINE:
            text = sentence.text
            y = self.pipeline.predict([text])
            if self.multilabel:
                y = self.y_encoder.inverse_transform(y)
                y = list(y[0])
            else:
                y = list(y)
            sentence.add_labels(y)
