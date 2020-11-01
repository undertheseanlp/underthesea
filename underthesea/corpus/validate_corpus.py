import sys
from os import listdir

from underthesea.file_utils import DATASETS_FOLDER

SUPPORTED_CORPUS_TYPE = set(["TOKENIZE"])


def validate_corpus_exist(corpus_name):
    if corpus_name not in listdir(DATASETS_FOLDER):
        print(f"Corpus {corpus_name} is not in existed in {DATASETS_FOLDER}")
        sys.exit(1)


def validate_corpus_type(corpus_type):
    if corpus_type not in SUPPORTED_CORPUS_TYPE:
        print(f"{corpus_type} is not supported")
        sys.exit(1)
    print("type")


def validate_corpus(corpus_type, corpus_name):
    print(f"Validate {corpus_type} corpus: {corpus_name}")
    validate_corpus_type(corpus_type)
    validate_corpus_content(corpus_name)
