import os
import re
import shutil
from enum import Enum
from typing import Union, List

from tabulate import tabulate

from underthesea.data_lf import CategorizedCorpus, Sentence, Corpus, Label
from underthesea.datasets_lf import REPO
from underthesea.file_utils import cached_path, CACHE_ROOT
from pathlib import Path
import zipfile

MISS_URL_ERROR = "Caution:\n  With closed license dataset, you must provide URL to download"
SAMPLE_CACHE_ROOT = Path(__file__).parent.absolute() / "data"


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
    def download_data(data, url):
        if data not in REPO:
            print(f"No matching distribution found for '{data}'")
            return

        filepath = REPO[data]["filepath"]
        cache_dir = REPO[data]["cache_dir"]
        filepath = Path(CACHE_ROOT) / cache_dir / filepath
        if Path(filepath).exists():
            print(f"Data is already existed: '{data}' in {filepath}")
            return

        if data == "VNESES":
            url = "https://www.dropbox.com/s/m4agkrbjuvnq4el/VNESEcorpus.txt?dl=1"
            cached_path(url, cache_dir=cache_dir)
            shutil.move(Path(CACHE_ROOT) / cache_dir / "VNESEcorpus.txt?dl=1",
                        Path(CACHE_ROOT) / cache_dir / filepath)

        if data == "VNTQ_SMALL":
            url = "https://www.dropbox.com/s/b0z17fa8hm6u1rr/VNTQcorpus-small.txt?dl=1"
            cached_path(url, cache_dir=cache_dir)
            shutil.move(Path(CACHE_ROOT) / cache_dir / "VNTQcorpus-small.txt?dl=1",
                        Path(CACHE_ROOT) / cache_dir / filepath)

        if data == "VNTQ_BIG":
            url = "https://www.dropbox.com/s/t4z90vs3qhpq9wg/VNTQcorpus-big.txt?dl=1"
            cached_path(url, cache_dir=cache_dir)
            shutil.move(Path(CACHE_ROOT) / cache_dir / "VNTQcorpus-big.txt?dl=1",
                        Path(CACHE_ROOT) / cache_dir / filepath)

        if data == "VNTC":
            url = "https://www.dropbox.com/s/4iw3xtnkd74h3pj/VNTC.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VNTC.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2013-WTK":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2013-WTK.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2013-POS":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2013-POS.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VTB-CHUNK":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VTB-CHUNK.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2016-NER":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2016-NER.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2018-NER":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2018-NER.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "AIVIVN2019_SA":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "AIVIVN2019_SA.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2016_SA":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2016_SA.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "VLSP2018_SA":
            if not url:
                print(f"\n{MISS_URL_ERROR}")
                return
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "VLSP2018_SA.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

        if data == "UTS2017_BANK":
            url = "https://www.dropbox.com/s/xl8sof2i1c35n62/UTS2017_BANK.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            filepath = Path(CACHE_ROOT) / cache_dir / "UTS2017_BANK.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(filepath)
            zip.extractall(cache_folder)
            os.remove(filepath)

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

        print(tabulate(datasets,
                       headers=["Name", "Type", "License", "Year", "Directory"],
                       tablefmt='orgtbl'))

        if all:
            print(f"\n{MISS_URL_ERROR}")

    @staticmethod
    def remove(data):
        if data not in REPO:
            print(f"No matching distribution found for '{data}'")
            return
        dataset = REPO[data]
        cache_dir = Path(CACHE_ROOT) / dataset["cache_dir"]
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        print(f"Dataset {data} is removed.")

    @staticmethod
    def load_corpus(corpus_id: Union[NLPData, str]) -> Corpus:
        if corpus_id == NLPData.AIVIVN2019_SA:
            data_folder = Path(CACHE_ROOT) / "datasets" / "aivivn2019_sa"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.AIVIVN2019_SA_SAMPLE:
            data_folder = SAMPLE_CACHE_ROOT / "aivivn2019_sa_sample"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.UTS2017_BANK_SA:
            data_folder = Path(CACHE_ROOT) / "datasets" / "uts2017_bank"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.UTS2017_BANK_SA_SAMPLE:
            data_folder = SAMPLE_CACHE_ROOT / "uts2017_bank_sample"
            return DataFetcher.load_classification_corpus(data_folder)

        if corpus_id == NLPData.UTS2017_BANK_TC:
            data_folder = Path(CACHE_ROOT) / "datasets" / "uts2017_bank"
            corpus = DataFetcher.load_classification_corpus(data_folder)
            return DataFetcher.__exact_aspect_labels(corpus)

        if corpus_id == NLPData.VLSP2016_SA:
            data_folder = Path(CACHE_ROOT) / "datasets" / "vlsp2016_sa"
            corpus = DataFetcher.load_classification_corpus(data_folder)
            return DataFetcher.__exact_aspect_labels(corpus)

        if corpus_id == NLPData.VNTC:
            data_folder = Path(CACHE_ROOT) / "datasets" / "VNTC"
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
        sentences_train: List[Sentence] = DataFetcher.read_text_classification_file(train_file)
        if dev_file.is_file():
            sentences_dev: List[Sentence] = DataFetcher.read_text_classification_file(dev_file)
        else:
            sentences_train, sentences_dev = DataFetcher.__sample(sentences_train)
        sentences_test: List[Sentence] = DataFetcher.read_text_classification_file(test_file)
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
            from underthesea.corpus_lf.vlsp2016_sa_corpus import VLSP2016SACorpus
            VLSP2016SACorpus.import_data(input_data_path)
