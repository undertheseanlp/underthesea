# -*- coding: utf-8 -*-
from os import listdir
from unittest import TestCase
from underthesea import pos_tag
from os.path import dirname, join

samples_dir = join(dirname(__file__), "samples")


def load_input(input_file):
    content = [text.split("\t")[0].decode("utf-8") for text in open(input_file, "r").read().strip().split("\n")]
    content = u" ".join(content)
    return content


def load_output(input_file):
    lines = [text.split("\t") for text in open(input_file, "r").read().strip().split("\n")]
    output = []
    for item in lines:
        word, tag = item
        output.append((word.decode("utf-8"), tag))
    return output


def save_temp(id, output):
    temp_file = join(samples_dir, "%s.correct" % id)
    content = u"\n".join([u"\t".join(item) for item in output])
    open(temp_file, "w").write(content.encode("utf-8"))


class TestPosTag(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = pos_tag(sentence)
        expected = []
        self.assertEqual(actual, expected)

    def test_accuracy(self):
        test_dir = join(dirname(__file__), "samples")
        files = listdir(test_dir)
        ids = [f.split(".")[0] for f in files]
        for id in ids:
            file = join(test_dir, "%s.txt" % id)
            sentence = load_input(file)
            actual = pos_tag(sentence)
            expected = load_output(file)
            if actual != expected:
                print("Fail {}".format(id))
                save_temp(id, actual)
            self.assertEqual(actual, expected)
