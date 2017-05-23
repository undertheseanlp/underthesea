# -*- coding: utf-8 -*-
from unittest import TestCase
from os import listdir
from os.path import dirname, join
import io
from tests.word_sent.test_config import EXPECTED_SPEED
from underthesea import word_sent
import time
from underthesea.word_sent.tokenize import tokenize


class TestPerformance(TestCase):
    def setUp(self):
        input_folder = join(dirname(__file__), "samples", "4_documents")
        files = listdir(input_folder)
        files = [join(input_folder, file) for file in files]
        texts = [io.open(file, "r", encoding="utf-8").read().split("\n") for file in files]
        texts = [text for sublist in texts for text in sublist]
        self.texts = texts

    def test_1(self):
        n_tokens = 0
        for text in self.texts:
            n_tokens += len(tokenize(text).split(" "))
        start = time.time()
        for text in self.texts:
            word_sent(text)
        end = time.time()
        duration = end - start  # in seconds
        speed = n_tokens / duration
        print "Speed: ", speed
        self.assertGreater(speed, EXPECTED_SPEED)

