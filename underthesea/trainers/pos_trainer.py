"""
POSTrainer - A user-friendly trainer for POS tagging.

This module provides a simplified interface for training POS taggers
using CRF models.

Example usage:
    from underthesea.trainers import POSTrainer
    from underthesea.datasets.vlsp2013_pos import VLSP2013_POS_SAMPLE

    corpus = VLSP2013_POS_SAMPLE()
    trainer = POSTrainer(corpus)
    trainer.train(output_dir='./models/pos', max_iterations=100)
"""
import logging
import os
import shutil
from os.path import join
from pathlib import Path
from typing import Union

import joblib
from seqeval.metrics import classification_report
from underthesea_core import CRFFeaturizer, CRFTagger
from underthesea_core import CRFTrainer as CoreCRFTrainer

from underthesea.transformer.tagged_feature import lower_words as dictionary

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)

# Default feature templates for POS tagging
DEFAULT_FEATURES = [
    # word unigram and bigram and trigram
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
    "T[-2,0]", "T[-1,1]", "T[0,2]",
    "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
    "T[0].istitle", "T[-1].istitle", "T[1].istitle",
]


def _read_tagged_data(data_file):
    """Read tagged corpus data from file.

    Args:
        data_file: Path to the data file.

    Returns:
        List of sentences, where each sentence is a list of [token, tag] pairs.
    """
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


def _preprocess_vlsp2013(dataset):
    """Preprocess VLSP2013 format by adding B- prefix to tags.

    Args:
        dataset: List of sentences.

    Returns:
        Preprocessed dataset with B- prefixed tags.
    """
    output = []
    for s in dataset:
        si = []
        for row in s:
            token, tag = row
            tag = "B-" + tag
            si.append([token, tag])
        output.append(si)
    return output


class POSTrainer:
    """
    A trainer for POS tagging models.

    This class provides a simplified interface for training POS taggers
    using CRF models with configurable feature templates.

    Args:
        corpus: A corpus object that provides train/dev/test file paths.
            Must have `train`, `dev`, and `test` attributes pointing to
            tagged corpus files in VLSP format.
        features (list, optional): List of feature templates.
            Default: DEFAULT_FEATURES (word n-grams and transformations).
        use_dictionary (bool): Whether to use Vietnamese dictionary features.
            Default: True.

    Example:
        >>> from underthesea.trainers import POSTrainer
        >>> from underthesea.datasets.vlsp2013_pos import VLSP2013_POS_SAMPLE
        >>> corpus = VLSP2013_POS_SAMPLE()
        >>> trainer = POSTrainer(corpus)
        >>> trainer.train(output_dir='./models/pos', max_iterations=100)
    """

    def __init__(
        self,
        corpus,
        features: list | None = None,
        use_dictionary: bool = True
    ):
        self.corpus = corpus
        self.features = features if features is not None else DEFAULT_FEATURES
        self.use_dictionary = use_dictionary

        # Build dictionary based on configuration
        self._dictionary = dictionary if use_dictionary else set()

        # Load and preprocess datasets
        self._train_dataset = None
        self._test_dataset = None

    def _load_datasets(self):
        """Load and preprocess training and test datasets."""
        if self._train_dataset is None:
            train_data = _read_tagged_data(self.corpus.train)
            self._train_dataset = _preprocess_vlsp2013(train_data)

        if self._test_dataset is None:
            test_data = _read_tagged_data(self.corpus.test)
            self._test_dataset = _preprocess_vlsp2013(test_data)

    def train(
        self,
        output_dir: Union[Path, str],
        max_iterations: int = 1000,
        c1: float = 1.0,
        c2: float = 1e-3,
    ):
        """
        Train the POS tagger.

        Args:
            output_dir (str or Path): Directory to save the trained model.
            max_iterations (int): Maximum number of CRF training iterations.
                Default: 1000.
            c1 (float): Coefficient for L1 penalty. Default: 1.0.
            c2 (float): Coefficient for L2 penalty. Default: 1e-3.

        Returns:
            None. The trained model is saved to `output_dir`.

        Example:
            >>> trainer.train(
            ...     output_dir='./models/pos',
            ...     max_iterations=100
            ... )
        """
        # Load datasets
        self._load_datasets()

        # Convert to Path and ensure it's a string for the trainer
        output_dir = str(Path(output_dir))

        # Create output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Create featurizer
        featurizer = CRFFeaturizer(self.features, self._dictionary)
        logger.info("Start feature extraction")

        X_train = featurizer.process(self._train_dataset)
        y_train = [[t[-1] for t in s] for s in self._train_dataset]
        y_test = [[t[-1] for t in s] for s in self._test_dataset]
        logger.info("Finish feature extraction")

        # Train
        logger.info("Start train")
        trainer = CoreCRFTrainer()
        trainer.set_l1_penalty(c1)
        trainer.set_l2_penalty(c2)
        trainer.set_max_iterations(max_iterations)

        model_path = join(output_dir, "models.bin")
        crf_model = trainer.train(X_train, y_train)
        crf_model.save(model_path)
        logger.info("Finish train")

        # Save features and dictionary
        joblib.dump(self.features, join(output_dir, "features.bin"))
        joblib.dump(self._dictionary, join(output_dir, "dictionary.bin"))

        # Evaluate on test set
        logger.info("Start evaluation")
        tagger = CRFTagger()
        tagger.load(model_path)
        y_pred = tagger.tag_batch(self._test_dataset, featurizer)

        # Write test output
        sentences = [[item[0] for item in sentence] for sentence in self._test_dataset]
        sentences = zip(sentences, y_test, y_pred)
        texts = []
        for s in sentences:
            tokens, y_true, y_pred_ = s
            tokens_ = []
            for i in range(len(tokens)):
                if tokens[i] == "":
                    token = "X"
                else:
                    token = tokens[i]
                tokens_.append(token + "\t" + y_true[i] + "\t" + y_pred_[i])
            text = "\n".join(tokens_)
            texts.append(text)
        text = "\n\n".join(texts)
        tmp_output_path = join(output_dir, "test_output.txt")
        open(tmp_output_path, "w").write(text)

        print("Classification report:\n")
        print(classification_report(y_test, y_pred, digits=3))
        logger.info("Finish evaluation")
