"""Tests for dependency parser training with PyTorch v2."""

import os
import shutil
import tempfile
import unittest

from underthesea.datasets.vlsp2020_dp import VLSP2020_DP_SAMPLE
from underthesea.models.dependency_parser import DependencyParser
from underthesea.modules.embeddings import CharacterEmbeddings, FieldEmbeddings
from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer


class TestDependencyParserTraining(unittest.TestCase):
    """Test dependency parser training pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, "test_dp_model")

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_train_sample_dataset(self):
        """Test training on sample dataset for 2 epochs."""
        # Load sample corpus
        corpus = VLSP2020_DP_SAMPLE()

        # Initialize parser with embeddings
        embeddings = [
            FieldEmbeddings(),
            CharacterEmbeddings()
        ]
        parser = DependencyParser(embeddings=embeddings, init_pre_train=True)

        # Create trainer and train
        trainer = DependencyParserTrainer(parser, corpus)
        trainer.train(
            base_path=self.model_path,
            max_epochs=2,
            mu=0.9
        )

        # Verify model was saved
        self.assertTrue(os.path.exists(self.model_path))

        # Load the trained model and make a prediction
        loaded_parser = DependencyParser.load(self.model_path)
        sentences = [["Tôi", "là", "sinh viên"]]
        dataset = loaded_parser.predict(sentences)

        # Verify predictions were made
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset.sentences), 1)


if __name__ == "__main__":
    unittest.main()
