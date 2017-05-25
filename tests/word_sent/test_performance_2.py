# -*- coding: utf-8 -*-
from unittest import TestCase
from os.path import dirname, join
import io
import time
from tests.word_sent.test_config import EXPECTED_SPEED
from underthesea.word_sent.tokenize import tokenize
from underthesea.word_sent.word_sent import word_sent


class TestPerformance2(TestCase):
    def setUp(self):
        file = join(dirname(__file__), "samples", "sentences", "448_tokens.txt")
        text = io.open(file, "r", encoding="utf-8").read()
        self.text = text

    def test_1(self):
        n_tokens = len(tokenize(self.text).split(" "))
        start = time.time()
        word_sent(self.text)
        end = time.time()
        duration = end - start  # in seconds
        if duration != 0:
            speed = n_tokens / duration
            print "Speed: ", speed
            self.assertGreater(speed, EXPECTED_SPEED)
