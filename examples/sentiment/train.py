import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import argparse
from preprocessing import ABSentimentData, Tokenizer4Bert


fnames = dict(
    restaurant=dict(
        train="/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Train.txt",
        test="/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Test.txt",
        dev="/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Dev.txt"
    ),
    hotel=dict(
        train="/Users/taidnguyen/Desktop/Sentence-level-Hotel/Train_Hotel.txt",
        test="/Users/taidnguyen/Desktop/Sentence-level-Hotel/Test_Hotel.txt",
        dev="/Users/taidnguyen/Desktop/Sentence-level-Hotel/Dev_Hotel.txt"
    )
)

# args
parser = argparse.ArgumentParser(description="Process some steps")
parser.add_argument("--train_path", type=str, default=fnames['restaurant']['test'])
parser.add_argument("--pretrained_bert_name", type=str, default="vinai/phobert-base")
parser.add_argument("--max_sequence_len", type=int, default=256)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=69)
args = parser.parse_args()


# load
data = ABSentimentData(args.train_path)
tokens = Tokenizer4Bert(args.max_sequence_len, args.pretrained_bert_name, data.sentences)
y = data.tones
X_train = tokens.get_features()
print(tokens)
print(X_train[0])
