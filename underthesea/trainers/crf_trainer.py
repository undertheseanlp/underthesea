import logging
import os
import shutil
from os.path import join

import pycrfsuite
from seqeval.metrics import classification_report
from underthesea_core import CRFFeaturizer

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)


class CRFTrainer:
    def __init__(self, model, training_args, train_dataset=None, test_dataset=None):
        self.model = model
        self.training_args = training_args
        self.output_dir = training_args["output_dir"]
        self.model_params = training_args["params"]
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self):
        # create output_dir directory
        output_dir = self.output_dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        features = self.model.features
        print(features)

        featurizer = CRFFeaturizer(features, set())
        logger.info("Start feature extraction")
        # featurizer = TaggedTransformer(features)
        # X_train, y_train = featurizer.transform(self.train_dataset, contain_labels=True)
        # X_test, y_test = featurizer.transform(self.test_dataset, contain_labels=True)

        X_train = featurizer.process(self.train_dataset)
        y_train = []
        for s in self.train_dataset:
            yi = [t[-1] for t in s]
            y_train.append(yi)
        X_test = featurizer.process(self.test_dataset)
        y_test = []
        for s in self.test_dataset:
            yi = [t[-1] for t in s]
            y_test.append(yi)
        logger.info("Finish feature extraction")

        # Train
        logger.info("Start train")
        trainer = pycrfsuite.Trainer(verbose=True)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        trainer.set_params(self.model_params)
        model_path = join(output_dir, "models.bin")
        trainer.train(model_path)
        logger.info("Finish train")
        self.model.save(output_dir)

        # Tagger
        logger.info("Start tagger")
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        y_pred = [tagger.tag(x_seq) for x_seq in X_test]
        sentences = [[item[0] for item in sentence] for sentence in self.test_dataset]
        sentences = zip(sentences, y_test, y_pred)
        texts = []
        for s in sentences:
            tokens, y_true, y_pred_ = s
            tokens_ = []
            for i in range(len(tokens)):
                if tokens[i] == "":
                    token = "X"
                else:
                    token = tokens[i]
                tokens_.append(token + "\t" + y_true[i] + "\t" + y_pred_[i])
            text = "\n".join(tokens_)
            texts.append(text)
        text = "\n\n".join(texts)
        tmp_output_path = join(output_dir, "test_output.txt")
        open(tmp_output_path, "w").write(text)
        print("Classification report:\n")
        print(classification_report(y_test, y_pred, digits=3))
        logger.info("Finish tagger")
