# -*- coding: utf-8 -*-
from os import listdir
import io
from unittest import TestCase

import sys

from underthesea import word_sent
from os.path import dirname, join


def load_input(input_file):
    text = io.open(input_file, "r", encoding="utf-8").read().split("\n")[0]

    if sys.version_info >= (3, 0):
        return text
    else:
        return text.decode("utf-8")


def load_output(output_file):
    if sys.version_info >= (3, 0):
        return [text for text in open(output_file, "r").read().strip().split("\n")]
    else:
        return [text.decode("utf-8") for text in open(output_file, "r").read().strip().split("\n")]


def save_temp(id, output):
    test_dir = join(dirname(__file__), "samples", "accuracy")
    temp_file = join(test_dir, "%s.tmp" % id)
    content = u"\n".join(output)
    if sys.version_info >= (3, 0):
        open(temp_file, "w", encoding="utf-8").write(content)
    else:
        open(temp_file, "w").write(content.encode("utf-8"))


class TestWord_sent(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = word_sent(sentence)
        expected = []
        self.assertEqual(actual, expected)

        actual = word_sent(sentence, format="text")
        expected = u""
        self.assertEqual(actual, expected)

    def test_word_sent(self):
        test_dir = join(dirname(__file__), "samples", "accuracy")
        files = listdir(test_dir)
        ids = [f.split(".")[0] for f in files if ".in" in f]
        for id in ids:
            input_file = join(test_dir, "%s.in" % id)
            output_file = join(test_dir, "%s.out" % id)
            sentence = load_input(input_file)
            actual = word_sent(sentence)
            expected = load_output(output_file)
            if actual != expected:
                print("Fail {}".format(id))
                save_temp(id, actual)
            self.assertEqual(actual, expected)
