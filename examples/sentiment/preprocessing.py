# Txt format:
# #1
# Mà không biết phải nhân viên cũ hay mới nữa nhưng cảm giác thân thiện hơn.
# {SERVICE#GENERAL, positive}
#
# #2
# Nay đi uống mới biết giá thành hơi cao nhưng thật sự đi đôi với chất lượng.
# {RESTAURANT#PRICES, negative}, {RESTAURANT#GENERAL, positive}
#

import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize

fnames = ["/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Dev.txt",
          "/Users/taidnguyen/Desktop/Sentence-level-Hotel/Dev_Hotel.txt"]


class Tokenizer4Bert:
    def __init__(self, max_sequence_len, pretrained_bert_name, sentences):
        self.model = AutoModel.from_pretrained(pretrained_bert_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name, use_fast=False)
        self.max_sequence_len = max_sequence_len

        for sentence in tqdm(sentences, "Tokenizing: "):
            word_segmented = word_tokenize(sentence, "text")
            self.token = self.tokenizer.encode(word_segmented)

    def get_features(self):
        """Pad, attention mask, and get features"""
        input_ids = torch.tensor([self.token])
        with torch.no_grad():
            features = self.model(input_ids)

        return features


class ABSentimentData:
    def __init__(self, fname):
        fin = open(fname, mode="r", encoding="utf-8", errors="ignore")
        lines = fin.readlines()
        fin.close()

        sentences = []
        aspects = []
        subaspects = []
        tones = []
        for i in tqdm(range(1, len(lines), 4), desc='Read data: '):
            sentence = lines[i].replace("\n", "")
            label_pattern = r"([\w#&]+)"  # keep joint aspects together ie. DESIGN&FEATURES
            labels = re.findall(label_pattern,
                                lines[i + 1].replace("\n", ""))  # array of 'AS#PECT', 'tone',... for sentence

            aspect = []
            tone = []
            for j in range(0, len(labels), 2):
                aspect.append(labels[j])
                tone.append(labels[j + 1])
                assert labels[j + 1] in ["positive", "negative", "neutral"], "Unexpected tone format:\n{}".format(
                    labels[j + 1])

            aspects.append(aspect)
            tones.append(tone)
            sentences.append(sentence)

        assert len(sentences) == len(aspects) == len(
            tones), "Mismatched data while splitting:\n{0}, {1}, {2}, {3}".format(len(sentences), len(aspects),
                                                                                  len(subaspects), len(tones))

        self.sentences = sentences
        self.aspects = aspects
        self.tones = tones

    def __getitem__(self, index):
        datum = {}
        datum["sentence"] = self.sentences[index]
        datum["aspects"] = self.aspects[index]
        datum["tones"] = self.tones[index]
        return datum

    def __len__(self):
        return len(self.sentences)
