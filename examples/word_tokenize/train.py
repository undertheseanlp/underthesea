from underthesea_core import CRFFeaturizer
from os.path import join
from pathlib import Path
import joblib
import pycrfsuite

# preparing training and testing data
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
from os.path import dirname
pwd = dirname(__file__)
full_train_dataset = load_dataset(join(pwd, "tmp/train.txt"))
full_test_dataset = load_dataset(join(pwd, "tmp/test.txt"))

print("Train Dataset:", len(full_train_dataset.X))
print("Test Dataset:", len(full_test_dataset.X))

import pycrfsuite
import logging
from underthesea_core import CRFFeaturizer
from underthesea.transformer.tagged_feature import lower_words as dictionary
from seqeval.metrics import classification_report
import shutil
import os

logger = logging.getLogger(__name__)
logger.setLevel(10)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)

class CRFTrainer:
    def __init__(self, model, training_args, train_dataset=None, test_dataset=None):
        self.model = model
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self):
        # create output_dir directory
        output_dir = self.training_args["output_dir"]
        if os.path.exists(output_dir):
          shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        logger.info("Start feature extraction")
        
        trainer = pycrfsuite.Trainer()
        count = 0
        for X_tokens, y_seq in zip(self.train_dataset.X[:10000], self.train_dataset.y[:10000]):
            X_seq = self.model.featurizer.process([X_tokens])[0]
            count += 1
            if count < 5:
              print(X_seq)
              len(X_seq)
              print(y_seq)
              len(y_seq)
            trainer.append(X_seq, y_seq)
        logger.info("Finish feature extraction")
        trainer.set_params(self.training_args["params"])
        
        filepath = join(output_dir, 'models.bin')
        # Train
        logger.info("Start train")
        trainer.train(filepath)
        
        self.model.save(output_dir)
        logger.info("Finish train")

        # Evaluation
        self.model.load(output_dir)
        logger.info("Start evaluation")
        y_pred = []
        for X in self.test_dataset.X:
            y_pred_ = self.model.predict([item[0] for item in X])
            y_pred.append(y_pred_)
            if "I-W" in y_pred_:
              print(y_pred)
        y_test = self.test_dataset.y
        
        print(classification_report(y_test, y_pred, digits=3))


#@title file fast_crf_sequence_tagger.py
from underthesea_core import CRFFeaturizer
from os.path import join
from pathlib import Path
import joblib
import pycrfsuite

class FastCRFSequenceTagger:
    def __init__(self, features=[], dictionary=set()):
        self.features = features
        self.dictionary = dictionary
        self.crf_tagger = None
        self.featurizer = CRFFeaturizer(self.features, self.dictionary)
        self.path_model = "models.bin"
        self.path_features = "features.bin"
        self.path_dictionary = "dictionary.bin"

    def forward(self, samples, contains_labels=False):
        return self.featurizer.transform(samples, contains_labels)

    def save(self, base_path):
        print("save features")
        joblib.dump(self.features, join(base_path, self.path_features))
        joblib.dump(self.dictionary, join(base_path, self.path_dictionary))

    def load(self, base_path):
        print(base_path)
        model_path = str(Path(base_path) / self.path_model)
        crf_tagger = pycrfsuite.Tagger()
        crf_tagger.open(model_path)
        features = joblib.load(join(base_path, self.path_features))
        dictionary = joblib.load(join(base_path, self.path_dictionary))
        featurizer = CRFFeaturizer(features, dictionary)
        self.featurizer = featurizer
        self.crf_tagger = crf_tagger

    def predict(self, tokens):
        tokens = [[token] for token in tokens]
        x = self.featurizer.process(tokens)
        # print(x)
        # print(len(x))
        # print(len(x[0]))
        tags = [self.crf_tagger.tag(item)[0] for item in x]
        return tags

#@title Train with FastCRFSequenceTagger 
from underthesea.transformer.tagged_feature import lower_words as dictionary

features = [
    "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",

    # "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",

    # "T[-1].istitle", "T[0].istitle", "T[1].istitle",
    # "T[0,1].istitle", "T[0,2].istitle",

    # "T[-2].is_in_dict", "T[-1].is_in_dict", "T[0].is_in_dict", "T[1].is_in_dict", "T[2].is_in_dict",
    # "T[-2,-1].is_in_dict", "T[-1,0].is_in_dict", "T[0,1].is_in_dict", "T[1,2].is_in_dict",
    # "T[-2,0].is_in_dict", "T[-1,1].is_in_dict", "T[0,2].is_in_dict",

    "T[-2,-1].lower", "T[-1,0].lower", "T[0,1].lower", "T[1,2].lower",
    # word unigram and bigram and trigram
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
    "T[-2,0]", "T[-1,1]", "T[0,2]",
]

model = FastCRFSequenceTagger(features, dictionary)

output_dir = 'tmp/fast_ws_20220219'
training_params = {
    'output_dir': output_dir,
    'params': {
      'c1': 1.0,  # coefficient for L1 penalty
      'c2': 1e-3,  # coefficient for L2 penalty
      'max_iterations': 100, 
      # include transitions that are possible, but not observed
      'feature.possible_transitions': True,
      'feature.possible_states': True,    
    }
}

train_dataset = full_train_dataset
test_dataset = full_test_dataset
trainer = CRFTrainer(model, training_params, train_dataset, test_dataset)

trainer.train()
