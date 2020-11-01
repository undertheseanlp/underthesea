from underthesea.corpus.validate_corpus import validate_corpus_exist


def revise_corpus_vlsp2013_wtk():
    source_corpus = "VLSP2013-WTK"
    validate_corpus_exist(source_corpus)
    version = 1
