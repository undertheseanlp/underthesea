import json
import warnings

# ignore warnings when using transformer
# see: https://github.com/scikit-learn/scikit-learn/issues/12327
from underthesea.corpus import Corpus
from underthesea.models.text_classifier import TextClassifier, TEXT_CLASSIFIER_ESTIMATOR
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from os.path import join
import joblib

warnings.simplefilter("ignore", category=PendingDeprecationWarning)


class ClassifierTrainer:

    def __init__(self, classifier: TextClassifier, corpus: Corpus):
        self.classifier = classifier
        self.corpus = corpus

    def train(self, model_folder: str, scoring=f1_score):
        score = {}
        multilabel = self.classifier.multilabel
        metadata = {"estimator": self.classifier.estimator.value, "multilabel": multilabel}
        train, dev, test = self._convert_corpus(self.corpus, multilabel=multilabel)
        X_train, y_train = train
        X_dev, y_dev = dev
        X_test, y_test = test
        # if self.classifier.estimator == TEXT_CLASSIFIER_ESTIMATOR.FAST_TEXT:
        #     hyper_params = self.classifier.params
        #     tmp_data_folder = tempfile.mkdtemp()
        #
        #     self.corpus.save(tmp_data_folder)
        #     train_file = Path(tmp_data_folder) / "train.txt"
        #     model = fastText.train_supervised(input=str(train_file), **hyper_params)
        #     y_dev_pred = [item[0].replace("__label__", "") for item in model.predict(X_dev)[0]]
        #     y_dev = [s.labels[0].value for s in self.corpus.dev]
        #     dev_score = scoring(y_dev, y_dev_pred)
        #
        #     y_test_pred = [item[0].replace("__label__", "") for item in model.predict(X_test)[0]]
        #     y_test = [s.labels[0].value for s in self.corpus.dev]
        #     test_score = scoring(y_test, y_test_pred)
        #
        #     shutil.rmtree(tmp_data_folder)
        #
        #     path_to_file = join(model_folder, "model.bin")
        #     model.save_model(path_to_file)
        #     print(f"Model is saved in {path_to_file}")
        #
        #     print("Dev score:", dev_score)
        #     print("Test score:", test_score)
        #     score["dev_score"] = dev_score
        #     score["test_score"] = test_score

        if self.classifier.estimator == TEXT_CLASSIFIER_ESTIMATOR.SVC:
            transformer = self.classifier.params['vectorizer']

            X_train = transformer.fit_transform(X_train)
            joblib.dump(transformer, join(model_folder, "x_transformer.joblib"))

            y_transformer = LabelEncoder()
            y_train = y_transformer.fit_transform(y_train)
            joblib.dump(y_transformer, join(model_folder, "y_transformer.joblib"))

            estimator = self.classifier.params['svc']
            estimator.fit(X_train, y_train)
            joblib.dump(estimator, join(model_folder, "estimator.joblib"))

            X_dev = transformer.transform(X_dev)
            y_dev = y_transformer.transform(y_dev)
            y_dev_pred = estimator.predict(X_dev)
            dev_score = scoring(y_dev, y_dev_pred)

            X_test = transformer.transform(X_test)
            y_test = y_transformer.transform(y_test)
            y_test_pred = estimator.predict(X_test)
            test_score = scoring(y_test, y_test_pred)
            score["dev_score"] = dev_score
            score["test_score"] = test_score
            print("Dev score:", dev_score)
            print("Test score:", test_score)

        if self.classifier.estimator == TEXT_CLASSIFIER_ESTIMATOR.PIPELINE:
            pipeline = self.classifier.pipeline
            if self.classifier.multilabel:
                y_train = self.classifier.y_encoder.fit_transform(y_train)
                joblib.dump(self.classifier.y_encoder, join(model_folder, "y_encoder.joblib"))
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, join(model_folder, "pipeline.joblib"))

            y_dev_pred = pipeline.predict(X_dev)
            if self.classifier.multilabel:
                dev_score = scoring(self.classifier.y_encoder.transform(y_dev), y_dev_pred)
            else:
                dev_score = scoring(y_dev, y_dev_pred)

            y_test_pred = pipeline.predict(X_test)
            if self.classifier.multilabel:
                test_score = scoring(self.classifier.y_encoder.transform(y_test), y_test_pred)
            else:
                test_score = scoring(y_test, y_test_pred)
            score["dev_score"] = dev_score
            score["test_score"] = test_score
            print("Dev score:", dev_score)
            print("Test score:", test_score)

        with open(join(model_folder, "metadata.json"), "w") as f:
            content = json.dumps(metadata, ensure_ascii=False)
            f.write(content)
        return score

    def _convert_corpus(self, corpus: Corpus, multilabel=False):
        X_train = [s.text for s in corpus.train]
        X_dev = [s.text for s in corpus.dev]
        X_test = [s.text for s in corpus.test]
        if multilabel:
            y_train = [[label.value for label in s.labels] for s in corpus.train]
            y_dev = [[label.value for label in s.labels] for s in corpus.dev]
            y_test = [[label.value for label in s.labels] for s in corpus.test]
        else:
            y_train = [s.labels[0].value for s in corpus.train]
            y_dev = [s.labels[0].value for s in corpus.dev]
            y_test = [s.labels[0].value for s in corpus.test]
        return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)
