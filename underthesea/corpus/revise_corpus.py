import sys
from os import makedirs
from os.path import join
from shutil import rmtree, copyfile
from underthesea.file_utils import DATASETS_FOLDER

from underthesea.corpus.validate_corpus import validate_corpus_exist

SUPPORTED_CORPUS = set(["VLSP2013-WTK"])


def revise_corpus_vlsp2013_wtk():
    source_corpus = "VLSP2013-WTK"
    validate_corpus_exist(source_corpus)
    version = 1
    dest_corpus = f"{source_corpus}-R{version}"
    source_folder = join(DATASETS_FOLDER, source_corpus)
    dest_folder = join(DATASETS_FOLDER, dest_corpus)
    try:
        rmtree(dest_folder)
    except Exception as e:
        pass
    makedirs(dest_folder)
    copyfile(join(source_folder, "train.txt"), join(dest_folder, "train.txt"))
    copyfile(join(source_folder, "test.txt"), join(dest_folder, "test.txt"))


def revise_corpus(corpus_name):
    if corpus_name not in SUPPORTED_CORPUS:
        print(f"Corpus {corpus_name} is not supported")
        sys.exit(1)
    if corpus_name == "VLSP2013-WTK":
        revise_corpus_vlsp2013_wtk()
