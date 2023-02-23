from os.path import dirname, join
from data import DataReader
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea.trainers.crf_trainer import CRFTrainer
from underthesea.transformer.tagged_feature import lower_words as dictionary


features = [
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
    "T[-2].lower",
    "T[-1].lower",
    "T[0].lower",
    "T[1].lower",
    "T[2].lower",
    "T[-2,-1].lower",
    "T[-1,0].lower",
    "T[0,1].lower",
    "T[1,2].lower",
    "T[-1].isdigit",
    "T[0].isdigit",
    "T[1].isdigit",
    "T[-2].istitle",
    "T[-1].istitle",
    "T[0].istitle",
    "T[1].istitle",
    "T[2].istitle",
    "T[0,1].istitle",
    "T[0,2].istitle",
    "T[-2].is_in_dict",
    "T[-1].is_in_dict",
    "T[0].is_in_dict",
    "T[1].is_in_dict",
    "T[2].is_in_dict",
    "T[-2,-1].is_in_dict",
    "T[-1,0].is_in_dict",
    "T[0,1].is_in_dict",
    "T[1,2].is_in_dict",
    "T[-2,0].is_in_dict",
    "T[-1,1].is_in_dict",
    "T[0,2].is_in_dict",
]
model = FastCRFSequenceTagger(features, dictionary)

pwd = dirname(__file__)
output_dir = join(pwd, "tmp/ws_20220222")
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


corpus = DataReader.load_tagged_corpus(
    join(pwd, "tmp"), train_file="train.txt", test_file="test.txt"
)
# train_dataset = corpus.train[:10000]
train_dataset = corpus.train
test_dataset = corpus.test
print("Train dataset", len(train_dataset))
print("Test dataset", len(test_dataset))

trainer = CRFTrainer(model, training_params, train_dataset, test_dataset)

trainer.train()
