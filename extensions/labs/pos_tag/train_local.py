import os
import shutil
from os.path import join

import data
import hydra
import joblib
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from seqeval.metrics import classification_report
from underthesea_core import CRFFeaturizer, CRFTagger
from underthesea_core import CRFTrainer as CoreCRFTrainer

from underthesea.transformer.tagged_feature import lower_words as dictionary


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def train(cfg: DictConfig) -> None:
    wd = get_original_cwd()
    print(OmegaConf.to_yaml(cfg))

    corpus = data.DataReader.load_tagged_corpus(
        join(wd, "tmp/vlsp2013"), train_file="train.txt", test_file="test.txt"
    )

    train_dataset = corpus.train
    train_dataset = data.preprocess_vlsp2013(train_dataset)

    train_samples = cfg["dataset"]["train_samples"]
    if train_samples != 0 and train_samples != -1:
        train_dataset = corpus.train[:train_samples]
    test_dataset = corpus.test

    test_dataset = corpus.test
    test_dataset = data.preprocess_vlsp2013(test_dataset)

    include_test = cfg["dataset"]["include_test"]
    if include_test:
        train_dataset += test_dataset

    features = [
        # word unigram and bigram and trigram
        "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
        "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
        "T[-2,0]", "T[-1,1]", "T[0,2]",
        "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
        "T[0].istitle", "T[-1].istitle", "T[1].istitle",
    ]

    output_dir = join(wd, "tmp/pos_tag")

    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Create featurizer and extract features
    featurizer = CRFFeaturizer(features, dictionary)
    X_train = featurizer.process(train_dataset)
    y_train = [[t[-1] for t in s] for s in train_dataset]
    y_test = [[t[-1] for t in s] for s in test_dataset]

    # Train
    trainer = CoreCRFTrainer()
    trainer.set_l1_penalty(1.0)
    trainer.set_l2_penalty(1e-3)
    trainer.set_max_iterations(1000)

    model_path = join(output_dir, "models.bin")
    crf_model = trainer.train(X_train, y_train)
    crf_model.save(model_path)

    # Save features and dictionary
    joblib.dump(features, join(output_dir, "features.bin"))
    joblib.dump(dictionary, join(output_dir, "dictionary.bin"))

    # Evaluate
    tagger = CRFTagger()
    tagger.load(model_path)
    y_pred = tagger.tag_batch(test_dataset, featurizer)
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == '__main__':
    train()
