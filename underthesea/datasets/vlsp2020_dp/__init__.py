from underthesea.corpus import Corpus
from underthesea.file_utils import DATASETS_FOLDER


class VLSP2020_DP(Corpus):
    def __init__(self):
        VLSP2020_DP_FOLDER = f'{DATASETS_FOLDER}/VLSP2020-DP'
        train_file = f'{VLSP2020_DP_FOLDER}/train.txt'
        test_file = f'{VLSP2020_DP_FOLDER}/test.txt'
        self._train = train_file
        self._test = test_file
        self._dev = test_file
