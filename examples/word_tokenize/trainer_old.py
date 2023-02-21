import os

import pycrfsuite
import logging
from conlleval import evaluate_
from underthesea.transformer.tagged import TaggedTransformer

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
from os.path import dirname, join

model_path = join(dirname(__file__), "tmp", "model.tmp")
tmp_output_path = join(dirname(__file__), "tmp", "output.txt")

class Trainer:
    def __init__(self, tagger, corpus):
        self.tagger = tagger
        self.corpus = corpus

    def train(self, params):
        features = self.tagger.features
        print(features)
        transformer = TaggedTransformer(features)
        logger.info("Start feature extraction")
        X_train, y_train = transformer.transform(self.corpus.train, contain_labels=True)
        X_train, y_train = X_train[:10000], y_train[:10000]
        X_test, y_test = transformer.transform(self.corpus.test, contain_labels=True)
        logger.info("Finish feature extraction")

        # Train
        logger.info("Start train")
        trainer = pycrfsuite.Trainer(verbose=True)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        trainer.set_params(params)
        trainer.train(model_path)
        logger.info("Finish train")

        # Tagger
        logger.info("Start tagger")
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        y_pred = [tagger.tag(x_seq) for x_seq in X_test]
        sentences = [[item[0] for item in sentence] for sentence in self.corpus.test]
        sentences = zip(sentences, y_test, y_pred)
        texts = []
        for s in sentences:
            tokens, y_true, y_pred = s
            tokens_ = []
            for i in range(len(tokens)):
                if tokens[i] == "":
                    token = "X"
                else:
                    token = tokens[i]
            tokens_.append(token + "\t" + y_true[i] + "\t" + y_pred[i])
            text = "\n".join(tokens_)
            texts.append(text)
        text = "\n\n".join(texts)
        open(tmp_output_path, "w").write(text)
        evaluate_(tmp_output_path)
        logger.info("Finish tagger")