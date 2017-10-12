# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import classify
from os.path import dirname, join

from underthesea.feature_engineering.text import Text
from underthesea.util.file_io import read, write

samples_dir = join(dirname(__file__), "samples")


def load_input(input_file):
    lines = read(input_file).strip().split("\n")
    content = [line.split("\t")[0] for line in lines]
    content = u" ".join(content)
    return content


def load_output(filename):
    lines = [text.split("\t") for text in read(filename).strip().split("\n")]
    output = [tuple(item) for item in lines]
    return output


def save_temp(id, output):
    temp_file = join(samples_dir, "%s.correct" % id)
    content = u"\n".join([u"\t".join(item) for item in output])
    write(temp_file, content)


class TestClassify(TestCase):
    def test_classify_null_cases(self):
        sentence = u""
        actual = classify(sentence)
        expected = None
        self.assertEqual(actual, expected)

    def test_classify_simple_case(self):
        text = u"HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam 54"
        actual = classify(text)[0]
        expected = Text("The thao")
        self.assertEqual(actual, expected)

