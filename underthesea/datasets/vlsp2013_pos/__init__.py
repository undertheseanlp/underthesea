import shutil
from os.path import isdir

from underthesea.corpus import Corpus
from underthesea.data_fetcher import DataFetcher
from underthesea.file_utils import DATASETS_FOLDER


class VLSP2013_POS(Corpus):
    def __init__(self):
        VLSP2013_POS_FOLDER = f'{DATASETS_FOLDER}/VLSP2013-POS'
        train_file = f'{VLSP2013_POS_FOLDER}/train.txt'
        test_file = f'{VLSP2013_POS_FOLDER}/test.txt'
        self._train = train_file
        self._test = test_file
        self._dev = test_file


class VLSP2013_POS_SAMPLE(Corpus):
    name = "VLSP2013_POS_SAMPLE"
    folder = f'{DATASETS_FOLDER}/{name}'
    REPO_DATA = {
        "url": 'https://github.com/undertheseanlp/resources/releases/download/1.3.x/VLSP2013_POS_SAMPLE-1.0-alpha.zip',
        "url_filename": "VLSP2013_POS_SAMPLE-1.0-alpha.zip",
        "cache_dir": f'datasets/{name}',
    }

    @staticmethod
    def download_file():
        if isdir(VLSP2013_POS_SAMPLE.folder):
            return
        shutil.rmtree(VLSP2013_POS_SAMPLE.folder, ignore_errors=True)
        DataFetcher.download_zip_file_to_cache(VLSP2013_POS_SAMPLE.REPO_DATA)

    def __init__(self):
        self.download_file()

        train_file = f'{self.folder}/train.txt'
        dev_file = f'{self.folder}/dev.txt'
        test_file = f'{self.folder}/test.txt'
        self._train = train_file
        self._test = dev_file
        self._dev = test_file
