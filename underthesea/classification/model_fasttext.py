from os.path import join, dirname
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
import fasttext
from underthesea.util.file_io import write
import os

from underthesea.util.singleton import Singleton


class FastTextClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.estimator = None

    def fit(self, X, y, model_filename=None):
        """Fit FastText according to X, y

        Parameters:
        ----------
        X : list of text
            each item is a text
        y: list
           each item is either a label (in multi class problem) or list of
           labels (in multi label problem)
        """
        train_file = "temp.train"
        X = [x.replace("\n", " ") for x in X]
        y = [item[0] for item in y]
        y = [_.replace(" ", "-") for _ in y]
        lines = ["__label__{} , {}".format(j, i) for i, j in zip(X, y)]
        content = "\n".join(lines)
        write(train_file, content)
        if model_filename:
            self.estimator = fasttext.supervised(train_file, model_filename)
        else:
            self.estimator = fasttext.supervised(train_file)
        os.remove(train_file)

    def predict(self, X):
        return

    def predict_proba(self, X):
        output_ = self.estimator.predict_proba(X)

        def transform_item(item):
            label, score = item[0]
            label = label.replace("__label__", "")
            label = int(label)
            if label == 0:
                label = 1
                score = 1 - score
            return [label, score]

        output_ = [transform_item(item) for item in output_]
        output1 = np.array(output_)
        return output1


@Singleton
class FastTextPredictor:
    def __init__(self):
        filepath = join(dirname(__file__), "fasttext.model")
        self.estimator = fasttext.load_model(filepath)

    def tranform_output(self, y):
        y = y[0].replace("__label__", "")
        y = y.replace("-", " ")
        return y

    def predict(self, X):
        X = [X]
        y_pred = self.estimator.predict(X)
        y_pred = [self.tranform_output(item) for item in y_pred]
        return y_pred
