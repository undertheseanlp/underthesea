import os
import re
import shutil
from enum import Enum
from os.path import dirname, join
from typing import Union, List
import yaml
from underthesea.corpus import Corpus
from underthesea.corpus.categorized_corpus import CategorizedCorpus
from underthesea.corpus.data import Sentence, Label
from underthesea.file_utils import cached_path, UNDERTHESEA_FOLDER
from pathlib import Path
import zipfile

from underthesea.utils import print_table

MISS_URL_ERROR = (
    "Caution:\n  With closed license dataset, you must provide URL to download"
)
SAMPLE_CACHE_ROOT = Path(__file__).parent.absolute() / "data"

CD = dirname(__file__)
with open(join(CD, "datasets.yaml")) as f:
    REPO = yaml.safe_load(f)


class NLPData(Enum):
    AIVIVN2019_SA = "aivivn2019_sa"
    AIVIVN2019_SA_SAMPLE = "aivivn2019_sa_sample"
    UTS2017_BANK_SA = "uts2017_bank_sa"
    UTS2017_BANK_TC = "uts2017_bank_tc"
    UTS2017_BANK_SA_SAMPLE = "uts2017_bank_sa_sample"
    VLSP2016_SA = "vlsp2016_sa"
    VNTC = "VNTC"


class DataFetcher:
    @staticmethod
    def download_raw_file_to_cache(repo_data):
        url = repo_data["url"]
        url_filename = repo_data["url_filename"]
        cache_dir = repo_data["cache_dir"]
        filepath = repo_data["filepath"]
        cached_path(url, cache_dir=cache_dir)
        shutil.move(
            Path(UNDERTHESEA_FOLDER) / cache_dir / url_filename,
            Path(UNDERTHESEA_FOLDER) / cache_dir / filepath,
        )

    @staticmethod
    def download_zip_file_to_cache(repo_data):
        url = repo_data["url"]
        cache_dir = repo_data["cache_dir"]
        url_filename = repo_data["url_filename"]
        cached_path(url, cache_dir=cache_dir)
        filepath = Path(UNDERTHESEA_FOLDER) / cache_dir / url_filename
        cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
        with zipfile.ZipFile(filepath) as zip:
            zip.extractall(cache_folder)
        os.remove(filepath)

    @staticmethod
    def download_data(data, url):
        if data not in REPO:
            print(f"No matching distribution found for '{data}'")
            return
        repo_data = REPO[data]
        if "url" in repo_data:
            url = repo_data["url"]
        filepath = REPO[data]["filepath"]
        cache_dir = REPO[data]["cache_dir"]
        filepath = Path(UNDERTHESEA_FOLDER) / cache_dir / filepath
        if Path(filepath).exists():
            print(f"Resource {data} is already existed in: {filepath}")
            return

        if data in set(["VNESES", "VNTQ_SMALL", "VNTQ_BIG"]):
            DataFetcher.download_raw_file_to_cache(repo_data)

        zip_datasets = [
            "VNTC",
            "VLSP2013-WTK",
            "VLSP2013-POS",
            "VTB-CHUNK",
            "VLSP2016-NER",
            "VLSP2018-NER",
            "AIVIVN2019-SA",
            "VLSP2016-SA",
            "VLSP2018-SA",
            "UTS2017-BANK",
            "DI_Vietnamese-UVD",
            "CP_Vietnamese-UNC",
            "SE_Vietnamese-UBS",
            "UIT_ABSA_RESTAURANT",
            "UIT_ABSA_HOTEL",
            "CP_Vietnamese_VLC_v2_2022",
        ]
        if data in set(zip_datasets):
            if repo_data["license"] == "Close":
                if not url:
                    print(f"\n{MISS_URL_ERROR}")
                    return
                repo_data["url"] = url
            DataFetcher.download_zip_file_to_cache(repo_data)
        print(f"Resource {data} is downloaded in {filepath} folder")

    @staticmethod
    def list(all):
        datasets = []
        for key in REPO:
            name = key
            type = REPO[key]["type"]
            license = REPO[key]["license"]
            year = REPO[key]["year"]
            directory = REPO[key]["cache_dir"]
            if not all:
                if license == "Close":
                    continue
            if license == "Close":
                license = "Close*"
            datasets.append([name, type, license, year, directory])
        datasets = list(
            sorted(datasets, key=lambda x: (x[3], x[1], x[0]), reverse=True)
        )
        print_table(datasets, headers=["Name", "Type", "License", "Year", "Directory"])

        if all:
            print(f"\n{MISS_URL_ERROR}")

    @staticmethod
    def remove(data):
        if data not in REPO:
            print(f"No matching distribution found for '{data}'")
            return
        dataset = REPO[data]
        cache_dir = Path(UNDERTHESEA_FOLDER) / dataset["cache_dir"]
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        print(f"Dataset {data} is removed.")

    @staticmethod
    def load_corpus(corpus_id: Union[NLPData, str]) -> Corpus:
        if corpus_id == NLPData.AIVIVN2019_SA:
            data_folder = Path(UNDERTHESEA_FOLDER) / "datasets" / "aivivn2019_sa"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.AIVIVN2019_SA_SAMPLE:
            data_folder = SAMPLE_CACHE_ROOT / "aivivn2019_sa_sample"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.UTS2017_BANK_SA:
            data_folder = Path(UNDERTHESEA_FOLDER) / "datasets" / "uts2017_bank"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.UTS2017_BANK_SA_SAMPLE:
            data_folder = SAMPLE_CACHE_ROOT / "uts2017_bank_sample"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.UTS2017_BANK_TC:
            data_folder = Path(UNDERTHESEA_FOLDER) / "datasets" / "uts2017_bank"
            corpus = DataFetcher.load_classification_corpus(data_folder)
            return DataFetcher.__exact_aspect_labels(corpus)

        if corpus_id == NLPData.VLSP2016_SA:
            data_folder = Path(UNDERTHESEA_FOLDER) / "datasets" / "vlsp2016_sa"
            corpus = DataFetcher.load_classification_corpus(data_folder)
            return DataFetcher.__exact_aspect_labels(corpus)

        if corpus_id == NLPData.VNTC:
            data_folder = Path(UNDERTHESEA_FOLDER) / "datasets" / "VNTC"
            corpus = DataFetcher.load_classification_corpus(data_folder)
            return DataFetcher.__exact_aspect_labels(corpus)

    @staticmethod
    def __exact_aspect_labels(corpus: CategorizedCorpus):
        def extract(data: List[Sentence]):
            for sentence in data:
                labels = []
                for label in sentence.labels:
                    value = label.value.split("#")[0]
                    label.value = value
                    labels.append(label)
                sentence.labels = labels
            return data

        train = extract(corpus.train)
        dev = extract(corpus.dev)
        test = extract(corpus.test)
        output_corpus = CategorizedCorpus(train, dev, test)
        return output_corpus

    @staticmethod
    def load_classification_corpus(data_folder) -> CategorizedCorpus:
        train_file = data_folder / "train.txt"
        dev_file = data_folder / "dev.txt"
        test_file = data_folder / "test.txt"
        sentences_train: List[Sentence] = DataFetcher.read_text_classification_file(
            train_file
        )
        if dev_file.is_file():
            sentences_dev: List[Sentence] = DataFetcher.read_text_classification_file(
                dev_file
            )
        else:
            sentences_train, sentences_dev = DataFetcher.__sample(sentences_train)
        sentences_test: List[Sentence] = DataFetcher.read_text_classification_file(
            test_file
        )
        corpus = CategorizedCorpus(sentences_train, sentences_dev, sentences_test)
        return corpus

    @staticmethod
    def __sample(data: List[Sentence], percentage: float = 0.1):
        import random

        random.shuffle(data)
        index = int(len(data) * percentage)
        a = data[:-index]
        b = data[-index:]
        return a, b

    @staticmethod
    def read_text_classification_file(path_to_file) -> List[Sentence]:
        sentences = []
        with open(path_to_file) as f:
            lines = f.read().splitlines()
            for line in lines:
                label_pattern = r"__label__(?P<label>[\w#]+)"
                labels = re.findall(label_pattern, line)
                labels = [Label(label) for label in labels]
                text = re.sub(label_pattern, "", line)
                s = Sentence(text, labels)
                sentences.append(s)
        return sentences

    @staticmethod
    # WIP
    def import_data(corpus_id: str, input_data_path: str):
        if corpus_id not in REPO:
            print(f"No matching distribution found for '{corpus_id}'")
            return
        if corpus_id == "VLSP2016_SA":
            from underthesea.datasets.vlsp2016_sa.vlsp2016_sa_corpus import (
                VLSP2016SACorpus,
            )

            VLSP2016SACorpus.import_data(input_data_path)
