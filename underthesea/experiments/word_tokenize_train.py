from underthesea.datasets.data import DataReader
from underthesea.datasets.vlsp2013_wtk_r2 import VLSP2013_WTK_R2
from underthesea.models.crf_sequence_tagger import CRFSequenceTagger
from underthesea.trainers import ModelTrainer

features = [
    # word unigram and bigram and trigram
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
    "T[-2,0]", "T[-1,1]", "T[0,2]",

    "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",
    "T[-2,-1].lower", "T[-1,0].lower", "T[0,1].lower", "T[1,2].lower",

    "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",

    "T[-2].istitle", "T[-1].istitle", "T[0].istitle", "T[1].istitle", "T[2].istitle",
    "T[0,1].istitle", "T[0,2].istitle",

    "T[-2].is_in_dict", "T[-1].is_in_dict", "T[0].is_in_dict", "T[1].is_in_dict", "T[2].is_in_dict",
    "T[-2,-1].is_in_dict", "T[-1,0].is_in_dict", "T[0,1].is_in_dict", "T[1,2].is_in_dict",
    "T[-2,0].is_in_dict", "T[-1,1].is_in_dict", "T[0,2].is_in_dict",
]
# corpus = VLSP2013_WTK_R2()
tagger = CRFSequenceTagger(features)
corpus = DataReader.load_tagged_corpus("/home/anhv/.underthesea/datasets/VLSP2013-WTK-R2",
                                       train_file="train.txt",
                                       test_file="test.txt")
trainer = ModelTrainer(tagger, corpus)

params = {
    'c1': 1.0,  # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 10000,  #
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True,
    'feature.possible_states': True,
}

trainer.train("models/wtk_crf", params)
