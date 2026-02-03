# -*- coding: utf-8 -*-
"""Tests for LinearSVC from underthesea_core."""

import os
import tempfile
from unittest import TestCase

import pytest

try:
    from underthesea_core import LinearSVC
    HAS_LINEAR_SVC = True
except ImportError:
    HAS_LINEAR_SVC = False


@pytest.mark.skipif(not HAS_LINEAR_SVC, reason="LinearSVC not available")
class TestLinearSVC(TestCase):
    """Test cases for LinearSVC classifier."""

    def test_create_empty_classifier(self):
        """Test creating an empty classifier."""
        svc = LinearSVC()
        self.assertEqual(svc.n_features, 0)
        self.assertEqual(len(svc.classes), 0)

    def test_fit_binary_classification(self):
        """Test binary classification."""
        svc = LinearSVC()

        # Simple 2D binary classification
        features = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
        labels = ["A", "A", "A", "B", "B", "B"]

        svc.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

        self.assertEqual(svc.n_features, 2)
        self.assertEqual(len(svc.classes), 2)
        self.assertIn("A", svc.classes)
        self.assertIn("B", svc.classes)

    def test_predict_single(self):
        """Test single sample prediction."""
        svc = LinearSVC()

        features = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        labels = ["X", "Y"]

        svc.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

        # Test prediction
        pred1 = svc.predict([0.9, 0.1])
        pred2 = svc.predict([0.1, 0.9])

        self.assertEqual(pred1, "X")
        self.assertEqual(pred2, "Y")

    def test_predict_batch(self):
        """Test batch prediction."""
        svc = LinearSVC()

        features = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        labels = ["A", "B", "C"]

        svc.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

        test_batch = [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
        ]

        predictions = svc.predict_batch(test_batch)

        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions[0], "A")
        self.assertEqual(predictions[1], "B")
        self.assertEqual(predictions[2], "C")

    def test_multiclass_classification(self):
        """Test multi-class classification."""
        svc = LinearSVC()

        # 4-class problem with distinct feature patterns
        features = []
        labels = []
        for i in range(4):
            for _ in range(10):
                f = [0.0] * 4
                f[i] = 1.0
                features.append(f)
                labels.append(f"class_{i}")

        svc.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

        self.assertEqual(len(svc.classes), 4)

        # Test predictions
        for i in range(4):
            test = [0.0] * 4
            test[i] = 1.0
            pred = svc.predict(test)
            self.assertEqual(pred, f"class_{i}")

    def test_save_load(self):
        """Test model serialization."""
        svc = LinearSVC()

        features = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        labels = ["P", "Q"]

        svc.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name

        try:
            svc.save(temp_path)

            # Load and verify
            loaded = LinearSVC.load(temp_path)

            self.assertEqual(loaded.n_features, svc.n_features)
            self.assertEqual(loaded.classes, svc.classes)

            # Same predictions
            test = [0.9, 0.1]
            self.assertEqual(loaded.predict(test), svc.predict(test))
        finally:
            os.unlink(temp_path)

    def test_sparse_features(self):
        """Test with sparse-like high dimensional features."""
        svc = LinearSVC()

        n_features = 1000
        features = []
        labels = []

        # Class A: features 0-10 active
        for _ in range(20):
            f = [0.0] * n_features
            for j in range(10):
                f[j] = 0.5 + (j * 0.05)
            features.append(f)
            labels.append("A")

        # Class B: features 500-510 active
        for _ in range(20):
            f = [0.0] * n_features
            for j in range(500, 510):
                f[j] = 0.5 + ((j - 500) * 0.05)
            features.append(f)
            labels.append("B")

        svc.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

        # Test with sparse-like input
        test_a = [0.0] * n_features
        test_a[5] = 1.0
        self.assertEqual(svc.predict(test_a), "A")

        test_b = [0.0] * n_features
        test_b[505] = 1.0
        self.assertEqual(svc.predict(test_b), "B")

    def test_vietnamese_text_labels(self):
        """Test with Vietnamese text labels."""
        svc = LinearSVC()

        features = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        labels = ["Thể thao", "Kinh tế", "Văn hóa"]

        svc.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

        pred = svc.predict([0.9, 0.05, 0.05])
        self.assertEqual(pred, "Thể thao")

        pred = svc.predict([0.05, 0.9, 0.05])
        self.assertEqual(pred, "Kinh tế")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
