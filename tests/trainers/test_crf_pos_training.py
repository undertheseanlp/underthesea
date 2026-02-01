"""Tests for POS tag training pipeline with CRFTrainer from underthesea_core."""

import os
import shutil
import tempfile
import unittest

import joblib
from underthesea_core import CRFFeaturizer, CRFTagger
from underthesea_core import CRFTrainer as CoreCRFTrainer


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


def read_tagged_data(data_file):
    """Read tagged corpus data from file."""
    sentences = []
    with open(data_file, encoding="utf-8") as f:
        raw_sentences = f.read().strip().split("\n\n")
    for s in raw_sentences:
        is_valid = True
        tagged_sentence = []
        for row in s.split("\n"):
            tokens = row.split("\t")
            tokens = [token.strip() for token in tokens]
            tagged_sentence.append(tokens)
        for row in tagged_sentence:
            if (len(row[0])) == 0:
                is_valid = False
        if is_valid:
            sentences.append(tagged_sentence)
    return sentences


def preprocess_vlsp2013(dataset):
    """Preprocess VLSP2013 format by adding B- prefix to tags."""
    output = []
    for s in dataset:
        si = []
        for row in s:
            token, tag = row
            tag = "B-" + tag
            si.append([token, tag])
        output.append(si)
    return output


class TestCRFPOSTrainingPipeline(unittest.TestCase):
    """Test POS training pipeline with CRFTrainer from underthesea_core."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, "test_pos_crf")
        cls.train_file = os.path.join(cls.temp_dir, "train.txt")
        cls.test_file = os.path.join(cls.temp_dir, "test.txt")

        # Write sample data
        with open(cls.train_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_POS_DATA)
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_POS_DATA)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_train_with_crf_trainer(self):
        """Test training POS tagger using CRFTrainer from underthesea_core."""
        # Load and preprocess data
        train_dataset = read_tagged_data(self.train_file)
        train_dataset = preprocess_vlsp2013(train_dataset)
        test_dataset = read_tagged_data(self.test_file)
        test_dataset = preprocess_vlsp2013(test_dataset)

        # Define features
        features = [
            "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
            "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
            "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
            "T[0].istitle", "T[-1].istitle", "T[1].istitle",
        ]
        dictionary = set()

        # Create output directory
        os.makedirs(self.model_path, exist_ok=True)

        # Create featurizer and extract features
        featurizer = CRFFeaturizer(features, dictionary)
        X_train = featurizer.process(train_dataset)
        y_train = [[t[-1] for t in s] for s in train_dataset]

        # Train using CRFTrainer from underthesea_core
        trainer = CoreCRFTrainer()
        trainer.set_l1_penalty(1.0)
        trainer.set_l2_penalty(1e-3)
        trainer.set_max_iterations(2)

        model_file = os.path.join(self.model_path, "models.bin")
        crf_model = trainer.train(X_train, y_train)
        crf_model.save(model_file)

        # Save features and dictionary
        joblib.dump(features, os.path.join(self.model_path, "features.bin"))
        joblib.dump(dictionary, os.path.join(self.model_path, "dictionary.bin"))

        # Verify model was saved
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(model_file))

        # Test loading and prediction
        tagger = CRFTagger()
        tagger.load(model_file)
        y_pred = tagger.tag_batch(test_dataset, featurizer)
        self.assertEqual(len(y_pred), len(test_dataset))


if __name__ == "__main__":
    unittest.main()
