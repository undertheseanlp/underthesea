from os.path import join, dirname
import os

# print(os.environ["PYTHONPATH"])
import data

pwd = dirname(__file__)

corpus = data.DataReader.load_tagged_corpus(
    join(pwd, "tmp/vlsp2013"), train_file="train.txt", test_file="test.txt"
)
train_dataset = corpus.train
train_dataset = data.preprocess_vlsp2013(train_dataset)
test_dataset = corpus.test
test_dataset = data.preprocess_vlsp2013(test_dataset)

#@title Training
from os.path import dirname, join
from underthesea.trainers.crf_trainer import CRFTrainer
from underthesea.transformer.tagged_feature import lower_words as dictionary
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger

features = [
    # word unigram and bigram and trigram
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
    "T[-2,0]", "T[-1,1]", "T[0,2]"
]
model = FastCRFSequenceTagger(features, dictionary)

pwd = dirname(__file__)
output_dir = join(pwd, "tmp/pos_tag")
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

# # Due to memory limit of Google Colab
# train_dataset = train_dataset[:10000]

trainer = CRFTrainer(model, training_params, train_dataset, test_dataset)

trainer.train()
