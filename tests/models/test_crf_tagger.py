# -*- coding: utf-8 -*-
"""
Unit tests for CRF components using underthesea_core.

Tests verify that CRFTagger and CRFTrainer from underthesea_core work correctly
as replacements for python-crfsuite.
"""
import tempfile
from pathlib import Path
from unittest import TestCase

from underthesea_core import CRFTagger, CRFTrainer


class TestCRFTrainerAndTagger(TestCase):
    """Test CRFTrainer and CRFTagger from underthesea_core."""

    def test_train_and_predict_simple(self):
        """Test basic training and prediction with simple data."""
        # Simple training data for BIO tagging
        X_train = [
            [["word=Hello", "is_upper=True"], ["word=world", "is_upper=False"]],
            [["word=Good", "is_upper=True"], ["word=morning", "is_upper=False"]],
        ]
        y_train = [
            ["B", "I"],
            ["B", "I"],
        ]

        # Train model
        trainer = CRFTrainer()
        trainer.set_l1_penalty(0.1)
        trainer.set_l2_penalty(0.01)
        trainer.set_max_iterations(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.crfsuite"
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            # Verify model file was created
            self.assertTrue(model_path.exists())

            # Load model and predict
            tagger = CRFTagger()
            tagger.load(str(model_path))

            # Test prediction
            X_test = [["word=Hello", "is_upper=True"], ["word=there", "is_upper=False"]]
            tags = tagger.tag(X_test)

            # Should return list of tags
            self.assertEqual(len(tags), 2)
            self.assertIn(tags[0], ["B", "I"])
            self.assertIn(tags[1], ["B", "I"])

    def test_train_pos_tagging_style(self):
        """Test training with POS tagging style features."""
        # POS-style training data
        X_train = [
            [
                ["word=Tôi", "lower=tôi", "is_title=True"],
                ["word=yêu", "lower=yêu", "is_title=False"],
                ["word=Việt", "lower=việt", "is_title=True"],
                ["word=Nam", "lower=nam", "is_title=True"],
            ],
            [
                ["word=Hà", "lower=hà", "is_title=True"],
                ["word=Nội", "lower=nội", "is_title=True"],
                ["word=đẹp", "lower=đẹp", "is_title=False"],
            ],
        ]
        y_train = [
            ["PRON", "VERB", "PROPN", "PROPN"],
            ["PROPN", "PROPN", "ADJ"],
        ]

        trainer = CRFTrainer()
        trainer.set_l1_penalty(0.1)
        trainer.set_l2_penalty(0.01)
        trainer.set_max_iterations(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "pos_model.crfsuite"
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            tagger = CRFTagger()
            tagger.load(str(model_path))

            # Test on training data (should get high accuracy on seen data)
            predictions = [tagger.tag(x) for x in X_train]
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(predictions[0]), 4)
            self.assertEqual(len(predictions[1]), 3)

    def test_tagger_batch_prediction(self):
        """Test batch prediction on multiple sequences."""
        X_train = [
            [["f1=a"], ["f1=b"]],
            [["f1=c"], ["f1=d"], ["f1=e"]],
        ]
        y_train = [
            ["X", "Y"],
            ["X", "Y", "X"],
        ]

        trainer = CRFTrainer()
        trainer.set_max_iterations(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "batch_model.crfsuite"
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            tagger = CRFTagger()
            tagger.load(str(model_path))

            # Batch prediction
            X_test = [
                [["f1=a"], ["f1=b"]],
                [["f1=x"], ["f1=y"], ["f1=z"]],
            ]
            predictions = [tagger.tag(x) for x in X_test]

            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(predictions[0]), 2)
            self.assertEqual(len(predictions[1]), 3)

    def test_empty_input(self):
        """Test handling of empty input."""
        X_train = [
            [["f=1"], ["f=2"]],
        ]
        y_train = [
            ["A", "B"],
        ]

        trainer = CRFTrainer()
        trainer.set_max_iterations(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "empty_test.crfsuite"
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            tagger = CRFTagger()
            tagger.load(str(model_path))

            # Empty sequence
            result = tagger.tag([])
            self.assertEqual(result, [])


class TestCRFTrainerParameters(TestCase):
    """Test CRFTrainer parameter settings."""

    def test_set_penalties(self):
        """Test setting L1 and L2 penalties."""
        trainer = CRFTrainer()
        trainer.set_l1_penalty(0.5)
        trainer.set_l2_penalty(0.05)
        # Should not raise any errors

    def test_set_max_iterations(self):
        """Test setting max iterations."""
        trainer = CRFTrainer()
        trainer.set_max_iterations(200)
        # Should not raise any errors


class TestCRFModelCompatibility(TestCase):
    """Test compatibility between CRFTrainer output and CRFTagger."""

    def test_model_save_load_cycle(self):
        """Test that models can be saved and loaded correctly."""
        X_train = [
            [["word=test", "pos=0"], ["word=data", "pos=1"]],
        ]
        y_train = [
            ["L1", "L2"],
        ]

        trainer = CRFTrainer()
        trainer.set_max_iterations(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "cycle_test.crfsuite"

            # Train and save
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            # Load with new tagger instance
            tagger1 = CRFTagger()
            tagger1.load(str(model_path))
            pred1 = tagger1.tag(X_train[0])

            # Load with another new tagger instance
            tagger2 = CRFTagger()
            tagger2.load(str(model_path))
            pred2 = tagger2.tag(X_train[0])

            # Both should produce same results
            self.assertEqual(pred1, pred2)


class TestOptimizedTagging(TestCase):
    """Test optimized tag_tokens and tag_batch methods."""

    def test_tag_tokens_with_featurizer(self):
        """Test tag_tokens method that combines featurization and tagging."""
        from underthesea_core import CRFFeaturizer

        # Train a model
        X_train = [
            [["word=Tôi", "pos=0"], ["word=yêu", "pos=1"]],
            [["word=Việt", "pos=0"], ["word=Nam", "pos=1"]],
        ]
        y_train = [
            ["B", "I"],
            ["B", "I"],
        ]

        trainer = CRFTrainer()
        trainer.set_max_iterations(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "opt_test.crfsuite"
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            # Create featurizer and tagger
            features = ["T[0][0]", "T[0][1]"]
            featurizer = CRFFeaturizer(features, set())
            tagger = CRFTagger()
            tagger.load(str(model_path))

            # Test tag_tokens (single sequence)
            tokens = [["Tôi", "X"], ["yêu", "Y"]]
            tags = tagger.tag_tokens(tokens, featurizer)

            self.assertEqual(len(tags), 2)
            self.assertIn(tags[0], ["B", "I"])
            self.assertIn(tags[1], ["B", "I"])

    def test_tag_batch_with_featurizer(self):
        """Test tag_batch method for batch processing."""
        from underthesea_core import CRFFeaturizer

        # Train a model
        X_train = [
            [["f1=a"], ["f1=b"]],
            [["f1=c"], ["f1=d"], ["f1=e"]],
        ]
        y_train = [
            ["X", "Y"],
            ["X", "Y", "X"],
        ]

        trainer = CRFTrainer()
        trainer.set_max_iterations(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "batch_opt_test.crfsuite"
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            # Create featurizer and tagger
            features = ["T[0][0]"]
            featurizer = CRFFeaturizer(features, set())
            tagger = CRFTagger()
            tagger.load(str(model_path))

            # Test tag_batch (multiple sequences)
            sequences = [
                [["a"], ["b"]],
                [["x"], ["y"], ["z"]],
            ]
            all_tags = tagger.tag_batch(sequences, featurizer)

            self.assertEqual(len(all_tags), 2)
            self.assertEqual(len(all_tags[0]), 2)
            self.assertEqual(len(all_tags[1]), 3)

    def test_tag_tokens_empty_sequence(self):
        """Test tag_tokens with empty sequence."""
        from underthesea_core import CRFFeaturizer

        X_train = [[["f=1"], ["f=2"]]]
        y_train = [["A", "B"]]

        trainer = CRFTrainer()
        trainer.set_max_iterations(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "empty_opt_test.crfsuite"
            model = trainer.train(X_train, y_train)
            model.save(str(model_path))

            features = ["T[0][0]"]
            featurizer = CRFFeaturizer(features, set())
            tagger = CRFTagger()
            tagger.load(str(model_path))

            # Empty sequence
            result = tagger.tag_tokens([], featurizer)
            self.assertEqual(result, [])
