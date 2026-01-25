"""Tests for POSTrainer - the user-friendly POS tagger trainer."""

import os
import shutil
import tempfile
import unittest


# Sample POS data in VLSP2013 format (token\tPOS_TAG)
SAMPLE_POS_DATA = """Tôi	P
yêu	V
Việt_Nam	Np
.	CH

Hôm_nay	N
trời	N
đẹp	A
.	CH

Anh	P
ấy	P
là	V
sinh_viên	N
.	CH
"""


class MockPOSCorpus:
    """Mock corpus for testing without network access."""

    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        self._train = os.path.join(temp_dir, "train.txt")
        self._test = os.path.join(temp_dir, "test.txt")
        self._dev = os.path.join(temp_dir, "test.txt")

        # Write sample data
        with open(self._train, "w", encoding="utf-8") as f:
            f.write(SAMPLE_POS_DATA)
        with open(self._test, "w", encoding="utf-8") as f:
            f.write(SAMPLE_POS_DATA)

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def dev(self):
        return self._dev


class TestPOSTrainer(unittest.TestCase):
    """Test POSTrainer functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, "test_pos_model")
        cls.corpus = MockPOSCorpus(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_train_sample_dataset(self):
        """Test training on sample dataset for 2 iterations."""
        from underthesea.trainers import POSTrainer

        trainer = POSTrainer(self.corpus)
        trainer.train(
            output_dir=self.model_path,
            max_iterations=2
        )

        # Verify model was saved
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(os.path.join(self.model_path, "models.bin")))
        self.assertTrue(os.path.exists(os.path.join(self.model_path, "features.bin")))
        self.assertTrue(os.path.exists(os.path.join(self.model_path, "dictionary.bin")))

    def test_trainer_initialization(self):
        """Test POSTrainer initialization with different configs."""
        from underthesea.trainers import POSTrainer
        from underthesea.trainers.pos_trainer import DEFAULT_FEATURES

        # Test default initialization
        trainer = POSTrainer(self.corpus)
        self.assertEqual(trainer.features, DEFAULT_FEATURES)
        self.assertTrue(trainer.use_dictionary)

        # Test with custom features
        custom_features = ["T[0]", "T[-1]", "T[1]"]
        trainer_custom = POSTrainer(self.corpus, features=custom_features)
        self.assertEqual(trainer_custom.features, custom_features)

        # Test without dictionary
        trainer_no_dict = POSTrainer(self.corpus, use_dictionary=False)
        self.assertFalse(trainer_no_dict.use_dictionary)


if __name__ == "__main__":
    unittest.main()
