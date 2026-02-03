# -*- coding: utf-8 -*-
"""Tests for TextClassifier, Label, and Sentence from underthesea_core."""

import os
import tempfile
from unittest import TestCase

import pytest

try:
    from underthesea_core import TextClassifier, Label, Sentence
    HAS_TEXT_CLASSIFIER = True
except ImportError:
    HAS_TEXT_CLASSIFIER = False


@pytest.mark.skipif(not HAS_TEXT_CLASSIFIER, reason="TextClassifier not available")
class TestLabel(TestCase):
    """Test cases for Label class."""

    def test_create_label(self):
        """Test creating a label."""
        label = Label("sports", 0.95)
        self.assertEqual(label.value, "sports")
        self.assertAlmostEqual(label.score, 0.95, places=2)

    def test_label_default_score(self):
        """Test label with default score."""
        label = Label("news")
        self.assertEqual(label.value, "news")
        self.assertEqual(label.score, 1.0)

    def test_label_score_clamping(self):
        """Test score clamping to [0, 1]."""
        label = Label("test", 1.5)
        self.assertEqual(label.score, 1.0)

        label2 = Label("test", -0.5)
        self.assertEqual(label2.score, 0.0)

    def test_label_str(self):
        """Test label string representation."""
        label = Label("sports", 0.85)
        s = str(label)
        self.assertIn("sports", s)
        self.assertIn("0.85", s)


@pytest.mark.skipif(not HAS_TEXT_CLASSIFIER, reason="TextClassifier not available")
class TestSentence(TestCase):
    """Test cases for Sentence class."""

    def test_create_sentence(self):
        """Test creating a sentence."""
        sentence = Sentence("Hello world")
        self.assertEqual(sentence.text, "Hello world")
        self.assertEqual(len(sentence.labels), 0)

    def test_sentence_with_labels(self):
        """Test sentence with labels."""
        labels = [Label("positive", 0.9)]
        sentence = Sentence("Great product!", labels)
        self.assertEqual(len(sentence.labels), 1)
        self.assertEqual(sentence.labels[0].value, "positive")

    def test_add_label(self):
        """Test adding a label to sentence."""
        sentence = Sentence("Test text")
        sentence.add_label(Label("category", 0.8))
        self.assertEqual(len(sentence.labels), 1)

    def test_add_labels(self):
        """Test adding multiple labels."""
        sentence = Sentence("Test text")
        sentence.add_labels([Label("cat1", 0.9), Label("cat2", 0.8)])
        self.assertEqual(len(sentence.labels), 2)


@pytest.mark.skipif(not HAS_TEXT_CLASSIFIER, reason="TextClassifier not available")
class TestTextClassifier(TestCase):
    """Test cases for TextClassifier."""

    def test_create_classifier(self):
        """Test creating a classifier."""
        clf = TextClassifier()
        self.assertFalse(clf.is_fitted)
        self.assertEqual(clf.n_features, 0)

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        clf = TextClassifier(max_features=1000, ngram_range=(1, 1), min_df=1)

        # Simple training data
        texts = [
            "thể thao bóng đá",
            "cầu thủ ghi bàn",
            "trận đấu bóng rổ",
            "kinh tế tài chính",
            "ngân hàng tiền tệ",
            "chứng khoán đầu tư",
        ]
        labels = ["sports", "sports", "sports", "business", "business", "business"]

        clf.fit(texts, labels)

        self.assertTrue(clf.is_fitted)
        self.assertGreater(clf.n_features, 0)

        # Test prediction
        pred = clf.predict("bóng đá việt nam")
        self.assertEqual(pred, "sports")

        pred2 = clf.predict("thị trường chứng khoán")
        self.assertEqual(pred2, "business")

    def test_predict_with_score(self):
        """Test predict with confidence score."""
        clf = TextClassifier(max_features=1000, ngram_range=(1, 1), min_df=1)

        texts = ["A B C", "A B D", "X Y Z", "X Y W"]
        labels = ["first", "first", "second", "second"]

        clf.fit(texts, labels)

        label, score = clf.predict_with_score("A B")
        self.assertEqual(label, "first")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_predict_batch(self):
        """Test batch prediction."""
        clf = TextClassifier(max_features=1000, ngram_range=(1, 1), min_df=1)

        texts = ["A B", "A C", "X Y", "X Z"]
        labels = ["cat1", "cat1", "cat2", "cat2"]

        clf.fit(texts, labels)

        test_texts = ["A B C", "X Y Z"]
        preds = clf.predict_batch(test_texts)

        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], "cat1")
        self.assertEqual(preds[1], "cat2")

    def test_predict_sentence(self):
        """Test predicting with Sentence object."""
        clf = TextClassifier(max_features=1000, ngram_range=(1, 1), min_df=1)

        texts = ["thể thao bóng đá", "kinh tế tài chính"]
        labels = ["sports", "business"]

        clf.fit(texts, labels)

        sentence = Sentence("bóng đá việt nam")
        clf.predict_sentence(sentence)

        self.assertEqual(len(sentence.labels), 1)
        self.assertEqual(sentence.labels[0].value, "sports")

    def test_save_load(self):
        """Test model serialization."""
        clf = TextClassifier(max_features=1000, ngram_range=(1, 1), min_df=1)

        texts = ["A B", "X Y"]
        labels = ["first", "second"]

        clf.fit(texts, labels)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name

        try:
            clf.save(temp_path)

            # Load and verify
            loaded = TextClassifier.load(temp_path)

            self.assertEqual(loaded.n_features, clf.n_features)
            self.assertEqual(loaded.classes, clf.classes)

            # Same predictions
            self.assertEqual(loaded.predict("A B C"), clf.predict("A B C"))
        finally:
            os.unlink(temp_path)

    def test_vietnamese_text(self):
        """Test with Vietnamese text."""
        clf = TextClassifier(max_features=5000, ngram_range=(1, 2), min_df=1)

        texts = [
            "Đội tuyển Việt Nam thắng Thái Lan 2-0",
            "Cầu thủ Quang Hải ghi bàn đẹp mắt",
            "Chứng khoán tăng mạnh hôm nay",
            "Ngân hàng công bố lãi suất mới",
        ]
        labels = ["Thể thao", "Thể thao", "Kinh doanh", "Kinh doanh"]

        clf.fit(texts, labels)

        pred = clf.predict("Trận đấu bóng đá hôm qua")
        self.assertIn(pred, ["Thể thao", "Kinh doanh"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
