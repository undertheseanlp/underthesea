from underthesea.corpus.conll_corpus import CONLLCorpus
from underthesea.file_utils import DATASETS_FOLDER
from underthesea.models.dependency_parser import DependencyParser
from underthesea.modules.embeddings import WordEmbedding, CharacterEmbedding
from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer


class VLSP2020_DP_R1(CONLLCorpus):
    def __init__(self):
        VLSP2020_DP_FOLDER = f'{DATASETS_FOLDER}/VLSP2020-DP-R1'
        train_file = f'{VLSP2020_DP_FOLDER}/train.txt'
        test_file = f'{VLSP2020_DP_FOLDER}/test.txt'
        super().__init__(train=train_file, test=test_file, dev=test_file)


class VLSP2020_DP_V1_0_0_A0(CONLLCorpus):
    def __init__(self):
        FOLDER = f'{DATASETS_FOLDER}/VLSP2020-DP-v1.0.0-a0'
        train_file = f'{FOLDER}/train.txt'
        dev_file = f'{FOLDER}/dev.txt'
        test_file = f'{FOLDER}/test.txt'
        super().__init__(train=train_file, test=test_file, dev=dev_file)


# corpus: CONLLCorpus = VLSP2020_DP_V1_0_0_A0()
corpus: CONLLCorpus = VLSP2020_DP_R1()

embeddings = [
    WordEmbedding(),
    CharacterEmbedding()
]
parser = DependencyParser(embeddings=embeddings, init_pre_train=True)
trainer: DependencyParserTrainer = DependencyParserTrainer(parser, corpus)
trainer.train(
    base_path='tmp/resources/parsers/dp-v1a1',
    max_epochs=1000,
    mu=.9  # optimizer parameters
)
