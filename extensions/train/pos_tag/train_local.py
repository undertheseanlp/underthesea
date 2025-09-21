from os.path import join

import data
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea.trainers.crf_trainer import CRFTrainer
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
    model = FastCRFSequenceTagger(features, dictionary)

    output_dir = join(wd, "tmp/pos_tag")
    training_params = {
        "output_dir": output_dir,
        "params": {
            "c1": 1.0,  # coefficient for L1 penalty
            "c2": 1e-3,  # coefficient for L2 penalty
            "max_iterations": 1000,  #
            # include transitions that are possible, but not observed
            "feature.possible_transitions": True,
            "feature.possible_states": True,
        },
    }

    trainer = CRFTrainer(model, training_params, train_dataset, test_dataset)

    trainer.train()


if __name__ == '__main__':
    train()
