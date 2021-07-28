import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import *

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


def main():
    entity = "hotel"

    # Args
    parser = argparse.ArgumentParser(description="Process some steps")
    parser.add_argument("--dev_path", type=str, default=fnames[entity]["dev"])
    parser.add_argument("--train_path", type=str, default=fnames[entity]["train"])
    parser.add_argument("--test_path", type=str, default=fnames[entity]["test"])
    parser.add_argument("--pretrained_bert_name", type=str, default="vinai/phobert-base")
    parser.add_argument("--max_sequence_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()

    seed_everything(args.seed)
    tokenizer = Tokenizer4Bert(args.max_sequence_len, args.pretrained_bert_name)

    # Train
    train = ABSADataset(args.dev_path, tokenizer)
    train_X, train_y = train.sentence_input_ids, train.aspect_input_ids
    # print(torch.tensor(train_X, dtype=torch.long))
    # print(torch.tensor(train_X, dtype=torch.long).shape)
    # print(torch.tensor(train_y, dtype=torch.long))
    # print(torch.tensor(train_y, dtype=torch.long).shape)


if __name__ == '__main__':
    main()






