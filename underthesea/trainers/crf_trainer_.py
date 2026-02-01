import logging

from underthesea_core import CRFFeaturizer, CRFTagger
from underthesea_core import CRFTrainer as CoreCRFTrainer

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
        trainer = CoreCRFTrainer()
        if "c1" in params:
            trainer.set_l1_penalty(params["c1"])
        if "c2" in params:
            trainer.set_l2_penalty(params["c2"])
        if "max_iterations" in params:
            trainer.set_max_iterations(params["max_iterations"])
        filename = 'tmp/model.tmp'
        model = trainer.train(X_train, y_train)
        model.save(filename)
        logger.info("Finish train")

        # Tagger
        logger.info("Start tagger")
        tagger = CRFTagger()
        tagger.load(filename)
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
