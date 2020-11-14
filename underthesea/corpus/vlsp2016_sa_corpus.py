import os
import shutil
from os.path import dirname, join

from underthesea.corpus.util import FolderStructure
from underthesea.file_utils import CACHE_ROOT
from pathlib import Path


def process_train_data(train_data_file, output_file, label):
    with open(train_data_file) as f:
        text = f.read()
        text = text.strip()
        sentences = text.split("\n\n")
    with open(output_file, "a") as f:
        for s in sentences:
            f.write(f"__label__{label} {s}\n")


class VLSP2016SACorpus:
    data_folder = Path(CACHE_ROOT) / "datasets" / "vlsp2016_sa"
    required_folder = True
    required_file = False

    @staticmethod
    def reset_folder():
        try:
            shutil.rmtree(VLSP2016SACorpus.data_folder)
        except Exception:
            pass
        os.makedirs(VLSP2016SACorpus.data_folder)

    @staticmethod
    def import_data(input_data_path: str):
        if VLSP2016SACorpus.required_folder and not Path(input_data_path).is_dir():
            print(f"Input path must be a folder but your input is {input_data_path}")
            raise SystemExit("")
        if VLSP2016SACorpus.required_file and not Path(input_data_path).is_file():
            print(f"Input path must be a file but your input is {input_data_path}")
            raise SystemExit("")
        raw_sample = join(dirname(dirname(__file__)), "data", "vlsp2016_sa_raw_sample")
        FolderStructure.check_structure(raw_sample, input_data_path)
        VLSP2016SACorpus.reset_folder()
        input_data_folder = Path(input_data_path)
        train_input_data_folder = input_data_folder / "SA2016-training_data"
        output_train_file = VLSP2016SACorpus.data_folder / "train.txt"
        with open(output_train_file, "w") as f:
            f.write("")
        process_train_data(train_input_data_folder / "SA-training_positive.txt", output_train_file, "POS")
        process_train_data(train_input_data_folder / "SA-training_neutral.txt", output_train_file, "NEU")
        process_train_data(train_input_data_folder / "SA-training_negative.txt", output_train_file, "NEG")

        # Preprocess Test Data
        test_sentences = []
        with open(input_data_folder / "SA2016-TestData-Ans" / "test_raw_ANS.txt", "r") as f:

            for i, line in enumerate(f):
                if i % 2 == 0:
                    text = line.strip()
                else:
                    label = line.strip()
                    sentence = f"__label__{label} {text}"
                    test_sentences.append(sentence)
        output_test_file = VLSP2016SACorpus.data_folder / "test.txt"
        with open(output_test_file, "w") as f:
            content = "\n".join(test_sentences)
            f.write(content + "\n")

        print(f"[LanguageFlow] Corpus `VLSP2016_SA` is imported in {VLSP2016SACorpus.data_folder}")
