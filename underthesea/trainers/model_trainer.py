from os import makedirs
from os.path import exists, join
from shutil import rmtree

from seqeval.metrics import accuracy_score, classification_report
from underthesea_core import CRFTagger
from underthesea_core import CRFTrainer as CoreCRFTrainer

from underthesea.file_utils import UNDERTHESEA_FOLDER
from underthesea.transformer.tagged import TaggedTransformer
from underthesea.utils import logger


class ModelTrainer:
    def __init__(self, tagger, corpus):
        self.tagger = tagger
        self.corpus = corpus

    def train(self, base_path, params):
        base_path = join(UNDERTHESEA_FOLDER, base_path)
        if exists(base_path):
            rmtree(base_path)
        makedirs(base_path)
        features = self.tagger.features
        print(features)
        transformer = TaggedTransformer(features)
        logger.info("Start feature extraction")
        train_samples = self.corpus.train
        test_samples = self.corpus.test
        X_train, y_train = transformer.transform(train_samples, contain_labels=True)
        X_test, y_test = transformer.transform(test_samples, contain_labels=True)
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
        filename = join(base_path, 'model.bin')
        model = trainer.train(X_train, y_train)
        model.save(filename)
        logger.info("Finish train")

        # Tagger
        logger.info("Start tagger")
        tagger = CRFTagger()
        tagger.load(filename)
        y_pred = [tagger.tag(x_seq) for x_seq in X_test]
        y_true = y_test
        print(classification_report(y_true, y_pred, digits=4))
        print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
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

        with open(join(base_path, "model.metadata"), "w") as f:
            f.write("model: CRFSequenceTagger")
        self.tagger.save(join(base_path, "features.bin"))

        logger.info("Finish tagger")
