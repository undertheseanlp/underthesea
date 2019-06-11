import os
import shutil
import zipfile
from enum import Enum
from tabulate import tabulate
from pathlib import Path

from underthesea.file_utls import cached_path, CACHE_ROOT
from underthesea.models import REPO


MISS_URL_ERROR = "Caution:\n  With closed license model, you must provide URL to download"


class UTSModel(Enum):
    se_svm_bank_2019_06 = 'se_svm_bank_2019_06'


class ModelFetcher:

    @staticmethod
    def download_model(model):
        if model not in REPO:
            print(f"No matching distribution found for '{model}'")
            return

        model_path = REPO[model]["model_path"]
        cache_dir = REPO[model]["cache_dir"]
        model_path = Path(CACHE_ROOT) / cache_dir / model_path
        if Path(model_path).exists():
            print(f"Model is already existed: '{model}' in {model_path}")
            return

        if model == "se_svm_bank_2019_06":
            url = "https://www.dropbox.com/s/vt80bre0a65kqco/se_svm_2019_06.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(CACHE_ROOT) / cache_dir / "se_svm_2019_06.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.remove(model_path)

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
    def remove(model):
        if model not in REPO:
            print(f"No matching distribution found for '{model}'")
            return
        model = REPO[model]
        cache_dir = Path(CACHE_ROOT) / model["cache_dir"]
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        print(f"Model {model} is removed.")

    @staticmethod
    def load_model(model):
        if model == UTSModel.se_svm_bank_2019_06:
            model_folder = Path(CACHE_ROOT) / "models" / "se_svm_bank_2019_06"
            return model_folder
