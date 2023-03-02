from omegaconf import DictConfig, OmegaConf
import hydra
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea.transformer.tagged_feature import lower_words as dictionary
from underthesea.trainers.crf_trainer import CRFTrainer
from os.path import dirname, join
from os.path import join, dirname
import data


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def train(cfg: DictConfig) -> None:
    pwd = dirname(__file__)
    corpus = data.DataReader.load_tagged_corpus(
        join(pwd, "tmp/vlsp2016_ner"), train_file="train.txt", test_file="test.txt"
    )
    
    train_dataset = corpus.train
    if "train_samples" in cfg["dataset"]:
        train_samples = cfg["dataset"]["train_samples"]
        if train_samples != 0:
            train_dataset = corpus.train[:train_samples]
    test_dataset = corpus.test

    features = [
        # word unigram and bigram and trigram
        "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
        "T[0].istitle", "T[-1].istitle", "T[1].istitle", "T[-2].istitle", "T[2].istitle",
        # word unigram and bigram
        "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
        "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
        # pos unigram and bigram
        "T[-2][1]", "T[-1][1]", "T[0][1]", "T[1][1]", "T[2][1]",
        "T[-2,-1][1]", "T[-1,0][1]", "T[0,1][1]", "T[1,2][1]",
    ]
    model = FastCRFSequenceTagger(features, dictionary)

    pwd = dirname(__file__)
    output_dir = join(pwd, "tmp/ner_20220303")
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
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train()
