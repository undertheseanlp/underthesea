import sys
import types
from unittest import TestCase
from unittest.mock import patch

from underthesea import ner


def _token(entity, word, start, end):
    return {"entity": entity, "word": word, "start": start, "end": end}


class TestNERDeepHyphen(TestCase):
    def _run(self, sentence, tokens):
        fake_module = types.ModuleType("model_transformers")
        fake_module.nlp = lambda _: [dict(token) for token in tokens]
        with patch.dict(sys.modules, {"underthesea.pipeline.ner.model_transformers": fake_module}):
            return ner(sentence, deep=True)

    def test_hyphenated_person_name(self):
        output = self._run("Jean-Luc Godard", [
            _token("B-PER", "Jean", 0, 4),
            _token("B-PER", "Luc", 5, 8),
            _token("I-PER", "Godard", 9, 15),
        ])
        self.assertEqual([item["entity"] for item in output], ["B-PER", "I-PER", "I-PER"])

    def test_hyphen_between_different_types_is_kept(self):
        output = self._run("Hà Nội-Putin", [
            _token("B-LOC", "Hà", 0, 2),
            _token("I-LOC", "Nội", 3, 6),
            _token("B-PER", "Putin", 7, 12),
        ])
        self.assertEqual([item["entity"] for item in output], ["B-LOC", "I-LOC", "B-PER"])

    def test_hyphenated_locations_are_kept_separate(self):
        output = self._run("Hà Nội-Hải Phòng", [
            _token("B-LOC", "Hà", 0, 2),
            _token("I-LOC", "Nội", 3, 6),
            _token("B-LOC", "Hải", 7, 10),
            _token("I-LOC", "Phòng", 11, 16),
        ])
        self.assertEqual([item["entity"] for item in output], ["B-LOC", "I-LOC", "B-LOC", "I-LOC"])

    def test_separate_entities_without_hyphen(self):
        output = self._run("Putin Trump", [
            _token("B-PER", "Putin", 0, 5),
            _token("B-PER", "Trump", 6, 11),
        ])
        self.assertEqual([item["entity"] for item in output], ["B-PER", "B-PER"])
