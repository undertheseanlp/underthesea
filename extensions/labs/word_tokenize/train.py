import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from os.path import join
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea.trainers.crf_trainer import CRFTrainer
from underthesea.transformer.tagged_feature import lower_words as dictionary
from datasets import load_dataset
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
    model = FastCRFSequenceTagger(features, dictionary)

    training_params = {
        "output_dir": join(wd, cfg.train.output_dir),
        "params": {
            "c1": cfg.train.params.c1,  # coefficient for L1 penalty
            "c2": cfg.train.params.c2,  # coefficient for L2 penalty
            "max_iterations": cfg.train.params.max_iterations,  #
            # include transitions that are possible, but not observed
            "feature.possible_transitions": cfg.train.params.feature.possible_transitions,
            "feature.possible_states": cfg.train.params.feature.possible_states,
        },
    }

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

    trainer = CRFTrainer(model, training_params, train_dataset, test_dataset)
    trainer.train()


if __name__ == "__main__":
    train()
