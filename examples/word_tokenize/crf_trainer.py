import logging
import os
import shutil
from os.path import join

import pycrfsuite
from seqeval.metrics import classification_report
from underthesea_core import CRFFeaturizer

from underthesea.transformer.tagged_feature import lower_words as dictionary

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)


class CRFTrainer:
    def __init__(self, model, training_args, train_dataset=None, test_dataset=None):
        self.model = model
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self):
        # create output_dir directory
        output_dir = self.training_args["output_dir"]
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        logger.info("Start feature extraction")

        trainer = pycrfsuite.Trainer()
        count = 0
        # loop through dataset

        for X_tokens, y_seq in zip(
            self.train_dataset.X[:10000], self.train_dataset.y[:10000]
        ):
            X_seq = self.model.featurizer.process([X_tokens])[0]
            count += 1
            if count < 5:
                print(X_seq)
                len(X_seq)
                print(y_seq)
                len(y_seq)
            trainer.append(X_seq, y_seq)
        logger.info("Finish feature extraction")
        trainer.set_params(self.training_args["params"])

        filepath = join(output_dir, "models.bin")
        # Train
        logger.info("Start train")
        trainer.train(filepath)

        self.model.save(output_dir)
        logger.info("Finish train")

        # Evaluation
        self.model.load(output_dir)

        logger.info("Start evaluation")
        y_pred = []
        for X in self.test_dataset.X:
            y_pred_ = self.model.predict([item[0] for item in X])
            y_pred.append(y_pred_)
            if "I-W" in y_pred_:
                print(y_pred)
        y_test = self.test_dataset.y

        print(classification_report(y_test, y_pred, digits=3))
