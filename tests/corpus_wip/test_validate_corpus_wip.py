from unittest import TestCase, skip
from underthesea.corpus.validate_corpus import validate_corpus


class TestValidateCorpus(TestCase):
    def test(self):
        corpus_name = "VLSP2013-WTK-R1"
        corpus_type = "TOKENIZE"
        validate_corpus(corpus_type, corpus_name)
