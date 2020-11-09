import sys
from underthesea.corpus.vlsp2013_wtk.revise_2 import revise_corpus_vlsp2013_wtk


SUPPORTED_CORPUS = set(["VLSP2013-WTK"])


def revise_corpus(corpus_name):
    if corpus_name not in SUPPORTED_CORPUS:
        print(f"Corpus {corpus_name} is not supported")
        sys.exit(1)
    if corpus_name == "VLSP2013-WTK":
        revise_corpus_vlsp2013_wtk()
