"""Tests for POS tag training pipeline with CRFTrainer."""

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


class TestPOSTrainingPipeline(unittest.TestCase):
    """Test POS training pipeline with CRFTrainer."""

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
        """Test training POS tagger using CRFTrainer directly."""
        from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
        from underthesea.trainers import CRFTrainer
        from underthesea.transformer.tagged_feature import lower_words as dictionary

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

        # Create model
        model = FastCRFSequenceTagger(features, dictionary)

        # Training parameters
        training_params = {
            "output_dir": self.model_path,
            "params": {
                "c1": 1.0,
                "c2": 1e-3,
                "max_iterations": 2,
                "feature.possible_transitions": True,
                "feature.possible_states": True,
            },
        }

        # Train
        trainer = CRFTrainer(model, training_params, train_dataset, test_dataset)
        trainer.train()

        # Verify model was saved
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(os.path.join(self.model_path, "models.bin")))


if __name__ == "__main__":
    unittest.main()
