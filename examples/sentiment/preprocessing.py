# UIT .txt data format:
# #1
# Mà không biết phải nhân viên cũ hay mới nữa nhưng cảm giác thân thiện hơn.
# {SERVICE#GENERAL, positive}
#
# #2
# Nay đi uống mới biết giá thành hơi cao nhưng thật sự đi đôi với chất lượng.
# {RESTAURANT#PRICES, negative}, {RESTAURANT#GENERAL, positive}
#

import re
import numpy as np
from tqdm import tqdm

from underthesea import word_tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def pad_and_truncate(sequence, max_len, dtype="int64", eos_id=2):
    """Pad or truncate BERT input ids accordingly
    End of sentence [SEP] == 2
    Do we need attention masks?
    """
    x = (np.ones(max_len) * 0).astype(dtype)  # init numpy array
    if len(sequence) > max_len:
        trunc = sequence[:max_len]
        trunc[-1] = eos_id
    else:
        trunc = sequence + [1] * (max_len - len(sequence))
    x[:len(trunc)] = trunc
    return x


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        self.object = RawDataParser(fname)

        sentences = self.object.sentences
        input_ids = []
        for i in tqdm(range(len(sentences)), "Getting input IDs for Dataset: "):
            sentence = sentences[i]
            input_id = tokenizer.get_input_id(sentence)
            input_ids.append(input_id)

        self.input_ids = input_ids


class Tokenizer4Bert:
    def __init__(self, max_sequence_len, pretrained_bert_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name, use_fast=False)
        self.max_sequence_len = max_sequence_len

    def get_input_id(self, text):
        word_segmented = word_tokenize(text, "text")  # PhoBert requires word_segmented text
        tokens = self.tokenizer.tokenize(word_segmented)
        input_id = self.tokenizer.convert_tokens_to_ids("[CLS]" + tokens)

        return pad_and_truncate(input_id, self.max_sequence_len)


class RawDataParser:
    def __init__(self, fname):
        print("Reading from %s" % fname)
        fin = open(fname, mode="r", encoding="utf-8", errors="ignore")
        lines = fin.readlines()
        fin.close()

        sentences = []
        aspects = []
        polarities = []
        for i in tqdm(range(1, len(lines), 4), desc="Read data: "):
            sentence = lines[i].replace("\n", "").replace(".", "")
            label_pattern = r"([\w#&]+)"  # keep joint aspects together ie. DESIGN&FEATURES
            labels = re.findall(label_pattern,
                                lines[i + 1].replace("\n", ""))  # [ASPECT', 'polarity',...] labels for single sentence

            aspect = []
            polarity = []
            for j in range(0, len(labels), 2):
                aspect.append(labels[j])
                polarity.append(labels[j + 1])
                assert labels[j + 1] in ["positive", "negative", "neutral"], "Unexpected polarity format:\n{}".format(
                    labels[j + 1])

            aspects.append(aspect)
            polarities.append(polarity)
            sentences.append(sentence)

        assert len(sentences) == len(aspects) == len(
            polarities), "Mismatched data while splitting:\n{0}, {1}, {2}".format(len(sentences), len(aspects),
                                                                                  len(polarities))

        self.sentences = sentences
        self.aspects = aspects
        self.polarities = polarities

    def __getitem__(self, index):
        datum = {
            "sentence": self.sentences[index],
            "aspects": self.aspects[index],
            "polarities": self.polarities[index]
        }
        return datum

    def __len__(self):
        return len(self.sentences)
