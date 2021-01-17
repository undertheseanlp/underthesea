import os
import shutil
import zipfile
from enum import Enum
from os.path import dirname, join
from pathlib import Path
import yaml
from underthesea.file_utils import cached_path, UNDERTHESEA_FOLDER
from underthesea.utils import print_table

MISS_URL_ERROR = "Caution:\n  With closed license model, you must provide URL to download"

CD = dirname(__file__)
with open(join(CD, "models.yaml")) as f:
    REPO = yaml.safe_load(f)


class UTSModel(Enum):
    tc_general = "TC_GENERAL"
    tc_bank = "TC_BANK"
    tc_general_v131 = "TC_GENERAL_V131"
    tc_bank_v131 = "TC_BANK_V131"
    sa_general = "SA_GENERAL"
    sa_bank = "SA_BANK"
    sa_bank_v131 = "SA_BANK_V131"

# flake8: noqa: C901
class ModelFetcher:

    @staticmethod
    def download(model_name):
        if model_name not in REPO:
            print(f"No matching distribution found for '{model_name}'")
            return

        model_path = REPO[model_name]["model_path"]
        cache_dir = REPO[model_name]["cache_dir"]
        model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / model_path
        if Path(model_path).exists():
            print(f"Model is already existed: '{model_name}' in {model_path}")
            return

        if model_name == "TC_GENERAL":
            url = "https://www.dropbox.com/s/866offu8wglrcej/tc_svm_vntc_20190607.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / "tc_svm_vntc_20190607.zip?dl=1"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / "tc_svm_vntc_20190607",
                Path(UNDERTHESEA_FOLDER) / cache_dir / "TC_GENERAL",
            )
            os.remove(model_path)

        if model_name == "TC_GENERAL_V131":
            url = "https://github.com/undertheseanlp/playground/releases/download/1.3.x/tc_svm_vntc_20201228.zip"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / "tc_svm_vntc_20201228.zip"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / "tc_svm_vntc_20201228",
                Path(UNDERTHESEA_FOLDER) / cache_dir / "TC_GENERAL_V131",
            )
            os.remove(model_path)

        if model_name == "TC_BANK_V131":
            model_id = "tc_svm_ubc-1_20210107"
            url = f"https://github.com/undertheseanlp/playground/releases/download/1.3.x/{model_id}.zip"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / f"{model_id}.zip"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / model_id,
                Path(UNDERTHESEA_FOLDER) / cache_dir / model_name,
            )
            os.remove(model_path)

        if model_name == "TC_BANK":
            url = "https://www.dropbox.com/s/prrjlypbrr6ze6p/tc_svm_uts2017_bank_20190607.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / "tc_svm_uts2017_bank_20190607.zip?dl=1"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / "tc_svm_uts2017_bank_20190607",
                Path(UNDERTHESEA_FOLDER) / cache_dir / "TC_BANK",
            )
            os.remove(model_path)

        if model_name == "SA_GENERAL":
            url = "https://www.dropbox.com/s/xfj1ity3egabv77/sa_svm_aivivn2019_20190615.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / "sa_svm_aivivn2019_20190615.zip?dl=1"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / "sa_svm_aivivn2019_20190615",
                Path(UNDERTHESEA_FOLDER) / cache_dir / "SA_GENERAL",
            )
            os.remove(model_path)

        if model_name == "SA_GENERAL_V131":
            model_id = "sa_svm_vlsp2016-sa_20210107"
            url = f"https://github.com/undertheseanlp/playground/releases/download/1.3.x/{model_id}.zip"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / f"{model_id}.zip"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / model_id,
                Path(UNDERTHESEA_FOLDER) / cache_dir / model_name,
            )
            os.remove(model_path)

        if model_name == "SA_BANK":
            url = "https://www.dropbox.com/s/yo6sf6ofpdb3hlh/sa_svm_uts2017_bank_20190611.zip?dl=1"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / "sa_svm_uts2017_bank_20190611.zip?dl=1"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / "sa_svm_uts2017_bank_20190611",
                Path(UNDERTHESEA_FOLDER) / cache_dir / "SA_BANK",
            )
            os.remove(model_path)

        if model_name == "SA_BANK_V131":
            model_id = "sa_svm_ubs-1_20210107"
            url = f"https://github.com/undertheseanlp/playground/releases/download/1.3.x/{model_id}.zip"
            cached_path(url, cache_dir=cache_dir)
            model_path = Path(UNDERTHESEA_FOLDER) / cache_dir / f"{model_id}.zip"
            cache_folder = Path(UNDERTHESEA_FOLDER) / cache_dir
            zip = zipfile.ZipFile(model_path)
            zip.extractall(cache_folder)
            os.rename(
                Path(UNDERTHESEA_FOLDER) / cache_dir / model_id,
                Path(UNDERTHESEA_FOLDER) / cache_dir / model_name,
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

        print_table(models, headers=["Name", "Type", "License", "Year", "Directory"])

        if all:
            print(f"\n{MISS_URL_ERROR}")

    @staticmethod
    def remove(model_name):
        if model_name not in REPO:
            print(f"No matching distribution found for '{model_name}'")
            return
        model = REPO[model_name]
        cache_dir = Path(UNDERTHESEA_FOLDER) / model["cache_dir"] / model["model_path"]
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        print(f"Model {model_name} is removed.")

    @staticmethod
    def get_model_path(model):
        if model == UTSModel.tc_bank:
            return Path(UNDERTHESEA_FOLDER) / "models" / "TC_BANK"

        if model == UTSModel.tc_general:
            return Path(UNDERTHESEA_FOLDER) / "models" / "TC_GENERAL"

        if model == UTSModel.sa_general:
            return Path(UNDERTHESEA_FOLDER) / "models" / "SA_GENERAL"

        if model == UTSModel.sa_bank:
            return Path(UNDERTHESEA_FOLDER) / "models" / "SA_BANK"
        return Path(UNDERTHESEA_FOLDER) / "models" / model
