from os.path import dirname
from crf_trainer import CRFTrainer
from fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea.transformer.tagged_feature import lower_words as dictionary
import logging
from os.path import join


class Dataset:
    def __init__(self):
        self.X = []
        self.y = []


def load_dataset(filepath) -> Dataset:
    dataset = Dataset()
    with open(filepath) as f:
        sentences = f.read().strip().split("\n\n")
        for sentence in sentences:
            rows = [row.split("\t") for row in sentence.split("\n")]
            Xs = [row[:-1] for row in rows if len(row) > 0]
            ys = [row[1] for row in rows if len(row) > 0]
            valid = True
            for i in range(len(Xs)):
                if len(Xs[i]) == 0 or len(ys[i]) == 0:
                    valid = False
            if len(Xs) > 0 and len(ys) > 0 and valid:
                dataset.X.append(Xs)
                dataset.y.append(ys)
    return dataset


pwd = dirname(__file__)
full_train_dataset = load_dataset(join(pwd, "tmp/train.txt"))
full_test_dataset = load_dataset(join(pwd, "tmp/test.txt"))

print("Train Dataset:", len(full_train_dataset.X))
print("Test Dataset:", len(full_test_dataset.X))


logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)


# @title Train with FastCRFSequenceTagger

features = [
    "T[-2].lower",
    "T[-1].lower",
    "T[0].lower",
    "T[1].lower",
    "T[2].lower",
    # "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",
    # "T[-1].istitle", "T[0].istitle", "T[1].istitle",
    # "T[0,1].istitle", "T[0,2].istitle",
    # "T[-2].is_in_dict", "T[-1].is_in_dict", "T[0].is_in_dict", "T[1].is_in_dict", "T[2].is_in_dict",
    # "T[-2,-1].is_in_dict", "T[-1,0].is_in_dict", "T[0,1].is_in_dict", "T[1,2].is_in_dict",
    # "T[-2,0].is_in_dict", "T[-1,1].is_in_dict", "T[0,2].is_in_dict",
    "T[-2,-1].lower",
    "T[-1,0].lower",
    "T[0,1].lower",
    "T[1,2].lower",
    # word unigram and bigram and trigram
    "T[-2]",
    "T[-1]",
    "T[0]",
    "T[1]",
    "T[2]",
    "T[-2,-1]",
    "T[-1,0]",
    "T[0,1]",
    "T[1,2]",
    "T[-2,0]",
    "T[-1,1]",
    "T[0,2]",
]

model = FastCRFSequenceTagger(features, dictionary)

output_dir = join(pwd, "tmp/fast_ws_20220219")
training_params = {
    "output_dir": output_dir,
    "params": {
        "c1": 1.0,  # coefficient for L1 penalty
        "c2": 1e-3,  # coefficient for L2 penalty
        "max_iterations": 100,
        # include transitions that are possible, but not observed
        "feature.possible_transitions": True,
        "feature.possible_states": True,
    },
}

train_dataset = full_train_dataset
test_dataset = full_test_dataset
trainer = CRFTrainer(model, training_params, train_dataset, test_dataset)

trainer.train()
