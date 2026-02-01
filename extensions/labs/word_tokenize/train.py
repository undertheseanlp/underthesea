import os
import shutil
from os.path import join

import hydra
import joblib
from datasets import load_dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from seqeval.metrics import classification_report
from underthesea_core import CRFFeaturizer, CRFTagger
from underthesea_core import CRFTrainer as CoreCRFTrainer

from underthesea.transformer.tagged_feature import lower_words as dictionary
from underthesea.utils.preprocess_dataset import preprocess_word_tokenize_dataset


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def train(cfg: DictConfig) -> None:
    wd = get_original_cwd()
    print(OmegaConf.to_yaml(cfg))

    features = [
        # word unigram and bigram and trigram
        "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
        "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]", "T[-2,0]",
        "T[-1,1]", "T[0,2]",
        "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
        "T[-2,-1].lower", "T[-1,0].lower", "T[0,1].lower", "T[1,2].lower",
        "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",
        "T[-2].istitle", "T[-1].istitle", "T[0].istitle", "T[1].istitle", "T[2].istitle",
        "T[0,1].istitle", "T[0,2].istitle",
        "T[-2].is_in_dict", "T[-1].is_in_dict", "T[0].is_in_dict", "T[1].is_in_dict", "T[2].is_in_dict",
        "T[-2,-1].is_in_dict", "T[-1,0].is_in_dict",
        "T[0,1].is_in_dict", "T[1,2].is_in_dict", "T[-2,0].is_in_dict",
        "T[-1,1].is_in_dict", "T[0,2].is_in_dict",
    ]

    output_dir = join(wd, cfg.train.output_dir)
    c1 = cfg.train.params.c1
    c2 = cfg.train.params.c2
    max_iterations = cfg.train.params.max_iterations

    dataset_name = cfg.dataset.name
    dataset_params = cfg.dataset.params

    # Check if subset exists in the config and load dataset accordingly
    if 'subset' in cfg.dataset:
        dataset_subset = cfg.dataset.subset
        dataset = load_dataset(dataset_name, dataset_subset, **dataset_params)
    else:
        dataset = load_dataset(dataset_name, **dataset_params)

    corpus = preprocess_word_tokenize_dataset(dataset)

    train_dataset = corpus["train"]
    test_dataset = corpus["test"]
    if cfg.dataset_extras.include_test:
        train_dataset = train_dataset + test_dataset
    print("Train dataset", len(train_dataset))
    print("Test dataset", len(test_dataset))

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
    trainer.set_l1_penalty(c1)
    trainer.set_l2_penalty(c2)
    trainer.set_max_iterations(max_iterations)

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


if __name__ == "__main__":
    train()
