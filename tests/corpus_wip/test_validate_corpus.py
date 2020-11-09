from unittest import TestCase
from underthesea.corpus.validate_corpus import validate_corpus


class TestValidateCorpus(TestCase):
    def test(self):
        corpus_name = "VLSP2013-WTK-R2"
        corpus_type = "TOKENIZE"
        validate_corpus(corpus_type, corpus_name, max_error=30000)
