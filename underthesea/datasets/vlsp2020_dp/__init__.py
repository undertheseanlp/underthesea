import shutil
from os.path import isdir

from underthesea.corpus import Corpus
from underthesea.data_fetcher import DataFetcher
from underthesea.file_utils import DATASETS_FOLDER


class VLSP2020_DP(Corpus):
    def __init__(self):
        VLSP2020_DP_FOLDER = f'{DATASETS_FOLDER}/VLSP2020-DP'
        train_file = f'{VLSP2020_DP_FOLDER}/train.txt'
        test_file = f'{VLSP2020_DP_FOLDER}/test.txt'
        self._train = train_file
        self._test = test_file
        self._dev = test_file


class VLSP2020_DP_SAMPLE(Corpus):
    name = "VLSP2020_DP_SAMPLE"
    folder = f'{DATASETS_FOLDER}/{name}'
    REPO_DATA = {
        "url": 'https://github.com/undertheseanlp/resources/releases/download/1.3.x/VLSP2020_DP_SAMPLE-1.0-alpha.zip',
        "url_filename": "VLSP2020_DP_SAMPLE-1.0-alpha.zip",
        "cache_dir": f'datasets/{name}',
    }

    @staticmethod
    def download_file():
        if isdir(VLSP2020_DP_SAMPLE.folder):
            return
        shutil.rmtree(VLSP2020_DP_SAMPLE.folder, ignore_errors=True)
        DataFetcher.download_zip_file_to_cache(VLSP2020_DP_SAMPLE.REPO_DATA)

    def __init__(self):
        self.download_file()

        train_file = f'{self.folder}/train.txt'
        dev_file = f'{self.folder}/dev.txt'
        test_file = f'{self.folder}/test.txt'
        self._train = train_file
        self._test = dev_file
        self._dev = test_file
