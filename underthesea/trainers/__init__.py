from os import makedirs
from os.path import join, exists
from shutil import rmtree

from underthesea.file_utils import CACHE_ROOT
from underthesea.trainers.conlleval import evaluate_
from underthesea.transformer.tagged import TaggedTransformer
import pycrfsuite
import logging

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)


class ModelTrainer:
    def __init__(self, tagger, corpus):
        self.tagger = tagger
        self.corpus = corpus

    def train(self, base_path, params):
        base_path = join(CACHE_ROOT, base_path)
        if exists(base_path):
            rmtree(base_path)
        makedirs(base_path)
        features = self.tagger.features
        print(features)
        transformer = TaggedTransformer(features)
        logger.info("Start feature extraction")
        X_train, y_train = transformer.transform(self.corpus.train, contain_labels=True)
        # X_train, y_train = X_train[:10000], y_train[:10000]
        X_test, y_test = transformer.transform(self.corpus.test, contain_labels=True)
        logger.info("Finish feature extraction")

        # Train
        logger.info("Start train")
        trainer = pycrfsuite.Trainer(verbose=True)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        trainer.set_params(params)
        filename = join(base_path, 'model.tmp')
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
        open(join(base_path, "output.txt"), "w").write(text)
        evaluate_(join(base_path, "output.txt"))
        logger.info("Finish tagger")

