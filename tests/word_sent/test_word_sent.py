# -*- coding: utf-8 -*-
from os import listdir
from unittest import TestCase
from underthesea import word_sent
from os.path import dirname, join
from underthesea.util.file_io import read, write


def load_input(input_file):
    text = read(input_file)
    text = text.split("\n")[0]
    return text


def load_output(output_file):
    return read(output_file).strip().split("\n")


def save_temp(id, output):
    test_dir = join(dirname(__file__), "samples", "accuracy")
    temp_file = join(test_dir, "%s.tmp" % id)
    content = u"\n".join(output)
    write(temp_file, content)


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
