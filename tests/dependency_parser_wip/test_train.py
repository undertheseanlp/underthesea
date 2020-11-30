from underthesea.file_utils import DATASETS_FOLDER
from underthesea.models.dependency_parser import BiaffineDependencyParser
from underthesea.trainers.parser_trainer import ParserTrainer


class Corpus:
    def __init__(self, train, dev=None, test=None, name: str = 'corpus'):
        self.name: str = name
        self._train = train
        self._dev = dev
        self._test = test

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test


class DPTagger:
    def __init__(self, name):
        self.name = name

    def tag(self, input, output):
        pass


class VLSP2020_DP_R1(Corpus):
    def __init__(self):
        VLSP2020_DP_FOLDER = f'{DATASETS_FOLDER}/VLSP2020-DP-R1'
        train_file = f'{VLSP2020_DP_FOLDER}/train.txt'
        test_file = f'{VLSP2020_DP_FOLDER}/test.txt'
        super().__init__(train=train_file, test=test_file, dev=test_file)


corpus: Corpus = VLSP2020_DP_R1()

embeddings = 'char'
embed = False
parser = BiaffineDependencyParser(embeddings, embed=False)
trainer = ParserTrainer(parser, corpus)
trainer.train(
    base_path='tmp/resources/parsers/dp',
    max_epochs=1,
    mu=0  # optimizer parameters
)
