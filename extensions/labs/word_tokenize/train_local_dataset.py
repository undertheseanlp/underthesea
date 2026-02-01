import os
import shutil
from os.path import dirname, join

import joblib
from data import DataReader
from seqeval.metrics import classification_report
from underthesea_core import CRFFeaturizer, CRFTagger
from underthesea_core import CRFTrainer as CoreCRFTrainer

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

pwd = dirname(__file__)
output_dir = join(pwd, "tmp/ws_20220222")

# Create output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

corpus = DataReader.load_tagged_corpus(
    join(pwd, "tmp"), train_file="train.txt", test_file="test.txt"
)
# train_dataset = corpus.train[:10000]
train_dataset = corpus.train
test_dataset = corpus.test
print("Train dataset", len(train_dataset))
print("Test dataset", len(test_dataset))

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
