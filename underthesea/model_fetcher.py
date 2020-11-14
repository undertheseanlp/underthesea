import os
import shutil
import zipfile
from enum import Enum
from os.path import dirname, join
from pathlib import Path
import yaml
from tabulate import tabulate
from underthesea.file_utils import cached_path, CACHE_ROOT

MISS_URL_ERROR = "Caution:\n  With closed license model, you must provide URL to download"

CD = dirname(__file__)
with open(join(CD, "models.yaml")) as f:
    REPO = yaml.safe_load(f)


class UTSModel(Enum):
    tc_general = "TC_GENERAL"
    tc_bank = "TC_BANK"

    sa_general = "SA_GENERAL"
    sa_bank = "SA_BANK"


class ModelFetcher:

    @staticmethod
    def download(model_name):
        if model_name not in REPO:
            print(f"No matching distribution found for '{model_name}'")
            return

        model_path = REPO[model_name]["model_path"]
        cache_dir = REPO[model_name]["cache_dir"]
        model_path = Path(CACHE_ROOT) / cache_dir / model_path
        if Path(model_path).exists():
            print(f"Model is already existed: '{model_name}' in {model_path}")
            return

        if model_name == "TC_GENERAL":
            url = "https://www.dropbox.com/s/866offu8wglrcej/tc_svm_vntc_20190607.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(CACHE_ROOT) / cache_dir / "tc_svm_vntc_20190607.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(CACHE_ROOT) / cache_dir / "tc_svm_vntc_20190607",
                Path(CACHE_ROOT) / cache_dir / "TC_GENERAL",
            )
            os.remove(model_path)

        if model_name == "TC_BANK":
            url = "https://www.dropbox.com/s/prrjlypbrr6ze6p/tc_svm_uts2017_bank_20190607.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(CACHE_ROOT) / cache_dir / "tc_svm_uts2017_bank_20190607.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(CACHE_ROOT) / cache_dir / "tc_svm_uts2017_bank_20190607",
                Path(CACHE_ROOT) / cache_dir / "TC_BANK",
            )
            os.remove(model_path)

        if model_name == "SA_GENERAL":
            url = "https://www.dropbox.com/s/xfj1ity3egabv77/sa_svm_aivivn2019_20190615.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(CACHE_ROOT) / cache_dir / "sa_svm_aivivn2019_20190615.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(CACHE_ROOT) / cache_dir / "sa_svm_aivivn2019_20190615",
                Path(CACHE_ROOT) / cache_dir / "SA_GENERAL",
            )
            os.remove(model_path)

        if model_name == "SA_BANK":
            url = "https://www.dropbox.com/s/yo6sf6ofpdb3hlh/sa_svm_uts2017_bank_20190611.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(CACHE_ROOT) / cache_dir / "sa_svm_uts2017_bank_20190611.zip?dl=1"
            cache_folder = Path(CACHE_ROOT) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(CACHE_ROOT) / cache_dir / "sa_svm_uts2017_bank_20190611",
                Path(CACHE_ROOT) / cache_dir / "SA_BANK",
            )
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
        if model == UTSModel.tc_bank:
            return Path(CACHE_ROOT) / "models" / "TC_BANK"

        if model == UTSModel.tc_general:
            return Path(CACHE_ROOT) / "models" / "TC_GENERAL"

        if model == UTSModel.sa_general:
            return Path(CACHE_ROOT) / "models" / "SA_GENERAL"

        if model == UTSModel.sa_bank:
            return Path(CACHE_ROOT) / "models" / "SA_BANK"
