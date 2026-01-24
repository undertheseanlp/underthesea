"""Tests for ParserTrainer - the user-friendly dependency parser trainer."""

import os
import shutil
import tempfile
import unittest

from underthesea.datasets.vlsp2020_dp import VLSP2020_DP_SAMPLE
from underthesea.models.dependency_parser import DependencyParser
from underthesea.trainers import ParserTrainer


class TestParserTrainer(unittest.TestCase):
    """Test ParserTrainer functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, "test_parser_model")

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_train_with_char_embeddings(self):
        """Test training with character-level embeddings (default)."""
        corpus = VLSP2020_DP_SAMPLE()

        trainer = ParserTrainer(corpus, feat='char')
        trainer.train(
            output_dir=self.model_path,
            max_epochs=2
        )

        # Verify model was saved
        self.assertTrue(os.path.exists(self.model_path))

        # Load and verify predictions
        loaded_parser = DependencyParser.load(self.model_path)
        sentences = [["Tôi", "yêu", "Việt Nam"]]
        dataset = loaded_parser.predict(sentences)

        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset.sentences), 1)

    def test_trainer_initialization(self):
        """Test ParserTrainer initialization with different configurations."""
        corpus = VLSP2020_DP_SAMPLE()

        # Test default initialization
        trainer = ParserTrainer(corpus)
        self.assertEqual(trainer.feat, 'char')
        self.assertIsNone(trainer.bert)

        # Test with tag features
        trainer_tag = ParserTrainer(corpus, feat='tag')
        self.assertEqual(trainer_tag.feat, 'tag')


if __name__ == "__main__":
    unittest.main()
