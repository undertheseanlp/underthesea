# -*- coding: utf-8 -*-
from os import listdir, mkdir
from unittest import TestCase, skip

import shutil

from underthesea import word_sent
from os.path import dirname, join, isfile
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
        ids = sorted([int(f.split(".")[0]) for f in files if ".in" in f])
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

    def _save_tmp_model(self, model_id, file_id, output):
        test_dir = join(dirname(__file__), "test_set", model_id)
        temp_file = join(test_dir, "%s.actual" % file_id)
        content = u"\n".join(output)
        write(temp_file, content)

    def _compare_model(self, id1, id2):
        fails1 = listdir(join(dirname(__file__), "test_set", id1))
        fails2 = listdir(join(dirname(__file__), "test_set", id2))
        new_fails = set(fails2) - set(fails1)
        new_improvements = set(fails1) - set(fails2)
        if len(new_fails):
            print("New fails:", new_fails)
        if len(new_improvements):
            print("New improvements:", new_improvements)

    def _test_model(self, model_id, model):
        name = model.__module__
        print("\n")
        print(name)
        test_dir = join(dirname(__file__), "test_set")
        files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
        ids = sorted([int(f.split(".")[0]) for f in files if ".in" in f])
        fails = []
        try:
            shutil.rmtree(join(test_dir, model_id))
        except:
            pass
        mkdir(join(test_dir, model_id))
        for id in ids:
            input_file = join(test_dir, "%s.in" % id)
            output_file = join(test_dir, "%s.out" % id)
            sentence = load_input(input_file)
            actual = model(sentence)
            expected = load_output(output_file)
            if actual != expected:
                fails.append(str(id))
                self._save_tmp_model(model_id, id, actual)
        if int(model_id) > 1:
            self._compare_model(str(int(model_id) -1), model_id)
        n = len(ids)
        correct = n - len(fails)
        print(
            "Accuracy: {:.2f}% ({}/{})".format(correct * 100.0 / n, correct, n))
        print("Fails   :", ", ".join(fails))

    @skip("")
    def test_models(self):
        models = [
            word_sent
        ]
        for i, model in enumerate(models):
            self._test_model(str(i + 1), model)
