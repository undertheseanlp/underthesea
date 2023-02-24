import pycrfsuite
import logging
from underthesea_core import CRFFeaturizer
from underthesea.transformer.tagged_feature import lower_words
# from conlleval import evaluate_

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)


class Trainer:
    def __init__(self, features, corpus):
        self.features = features
        self.corpus = corpus

    def train(self, params):
        transformer = CRFFeaturizer(self.features, lower_words)
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
        filename = 'tmp/model.tmp'
        trainer.train(filename)
        logger.info("Finish train")

        # Tagger
        logger.info("Start tagger")
        tagger = pycrfsuite.Tagger()
        tagger.open(filename)
        y_pred = [tagger.tag(x_seq) for x_seq in X_test]
        sentences = [[item[0] for item in sentence] for sentence in self.corpus.test]
        sentences = zip(sentences, y_test, y_pred)
        texts = []
        for s in sentences:
            tokens, y_true, y_pred = s
            tokens_ = ["\t".join(item) for item in zip(tokens, y_true, y_pred)]
            text = "\n".join(tokens_)
            texts.append(text)
        text = "\n\n".join(texts)
        open("tmp/output.txt", "w").write(text)
        # evaluate_("tmp/output.txt")
        logger.info("Finish tagger")
