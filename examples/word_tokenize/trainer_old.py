import logging
from os.path import dirname, join

import pycrfsuite
from seqeval.metrics import classification_report
from underthesea.transformer.tagged import TaggedTransformer

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)


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
        # print("y_pred\n", y_pred)
        # print("y_test\n", y_test)
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
        open(tmp_output_path, "w").write(text)
        print("Classification report:\n")
        print(classification_report(y_test, y_pred, digits=3))
        logger.info("Finish tagger")
