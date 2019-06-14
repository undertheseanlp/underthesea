import os
import shutil
import zipfile
from enum import Enum
from pathlib import Path

from tabulate import tabulate

from underthesea.file_utils import cached_path, CACHE_ROOT
from underthesea.models import REPO

MISS_URL_ERROR = "Caution:\n  With closed license model, you must provide URL to download"


class UTSModel(Enum):
    tc_svm_uts2017_bank_20190607 = "tc_svm_uts2017_bank_20190607"
    tc_svm_vntc_20190607 = "tc_svm_vntc_20190607"


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

        if model == "tc_svm_uts2017_bank_20190607":
            url = "https://www.dropbox.com/s/prrjlypbrr6ze6p/tc_svm_uts2017_bank_20190607.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(CACHE_ROOT) / cache_dir / "tc_svm_uts2017_bank_20190607.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.remove(model_path)

        if model == "tc_svm_vntc_20190607":
            url = "https://www.dropbox.com/s/866offu8wglrcej/tc_svm_vntc_20190607.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(CACHE_ROOT) / cache_dir / "tc_svm_vntc_20190607.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.remove(model_path)

    @staticmethod
    def list(all):
        models = []
        for key in REPO:
            name = key
            type = REPO[key]["type"]
            license = REPO[key]["license"]
            year = REPO[key]["year"]
            directory = Path(REPO[key]["cache_dir"]) / REPO[key]["model_path"]
            if not all:
                if license == "Close":
                    continue
            if license == "Close":
                license = "Close*"
            models.append([name, type, license, year, directory])

        print(tabulate(models,
                       headers=["Name", "Type", "License", "Year", "Directory"],
                       tablefmt='orgtbl'))

        if all:
            print(f"\n{MISS_URL_ERROR}")

    @staticmethod
    def remove(model_name):
        if model_name not in REPO:
            print(f"No matching distribution found for '{model_name}'")
            return
        model = REPO[model_name]
        cache_dir = Path(CACHE_ROOT) / model["cache_dir"] / model["model_path"]
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        print(f"Model {model_name} is removed.")

    @staticmethod
    def get_model_path(model):
        if model == UTSModel.tc_svm_uts2017_bank_20190607:
            model_folder = Path(CACHE_ROOT) / "models" / "tc_svm_uts2017_bank_20190607"
            return model_folder

        if model == UTSModel.tc_svm_vntc_20190607:
            return Path(CACHE_ROOT) / "models" / "tc_svm_vntc_20190607"
