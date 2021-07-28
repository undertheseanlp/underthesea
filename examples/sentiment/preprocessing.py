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
import torch
from tqdm import tqdm
from underthesea import word_tokenize
from transformers import AutoTokenizer, AutoModel

aspect2ids = {
    'restaurant': {'SERVICE#GENERAL': 0, 'RESTAURANT#PRICES': 1, 'RESTAURANT#GENERAL': 2, 'FOOD#STYLE&OPTIONS': 3, 'FOOD#QUALITY': 4, 'AMBIENCE#GENERAL': 5, 'RESTAURANT#MISCELLANEOUS': 6, 'LOCATION#GENERAL': 7, 'DRINKS#QUALITY': 8, 'FOOD#PRICES': 9, 'DRINKS#PRICES': 10, 'DRINKS#STYLE&OPTIONS': 11},
    'hotel': {'ROOMS#DESIGN&FEATURES': 0, 'ROOMS#CLEANLINESS': 1, 'FOOD&DRINKS#STYLE&OPTIONS': 2, 'LOCATION#GENERAL': 3, 'SERVICE#GENERAL': 4, 'ROOMS#QUALITY': 5, 'FOOD&DRINKS#QUALITY': 6, 'HOTEL#GENERAL': 7, 'ROOMS#PRICES': 8, 'ROOM_AMENITIES#GENERAL': 9, 'ROOMS#COMFORT': 10, 'FACILITIES#GENERAL': 11, 'ROOM_AMENITIES#DESIGN&FEATURES': 12, 'HOTEL#DESIGN&FEATURES': 13, 'HOTEL#COMFORT': 14, 'HOTEL#QUALITY': 15, 'ROOM_AMENITIES#QUALITY': 16, 'HOTEL#PRICES': 17, 'HOTEL#CLEANLINESS': 18, 'ROOM_AMENITIES#COMFORT': 19, 'HOTEL#MISCELLANEOUS': 20, 'ROOMS#GENERAL': 21, 'ROOM_AMENITIES#MISCELLANEOUS': 22, 'ROOM_AMENITIES#CLEANLINESS': 23, 'FACILITIES#QUALITY': 24, 'FACILITIES#DESIGN&FEATURES': 25, 'FOOD&DRINKS#MISCELLANEOUS': 26, 'FACILITIES#CLEANLINESS': 27, 'FACILITIES#COMFORT': 28, 'FACILITIES#MISCELLANEOUS': 29, 'FOOD&DRINKS#PRICES': 30, 'FACILITIES#PRICES': 31, 'ROOMS#MISCELLANEOUS': 32}
}
id2aspects = {
    'restaurant': {0: 'SERVICE#GENERAL', 1: 'RESTAURANT#PRICES', 2: 'RESTAURANT#GENERAL', 3: 'FOOD#STYLE&OPTIONS', 4: 'FOOD#QUALITY', 5: 'AMBIENCE#GENERAL', 6: 'RESTAURANT#MISCELLANEOUS', 7: 'LOCATION#GENERAL', 8: 'DRINKS#QUALITY', 9: 'FOOD#PRICES', 10: 'DRINKS#PRICES', 11: 'DRINKS#STYLE&OPTIONS'},
    'hotel': {0: 'ROOMS#DESIGN&FEATURES', 1: 'ROOMS#CLEANLINESS', 2: 'FOOD&DRINKS#STYLE&OPTIONS', 3: 'LOCATION#GENERAL', 4: 'SERVICE#GENERAL', 5: 'ROOMS#QUALITY', 6: 'FOOD&DRINKS#QUALITY', 7: 'HOTEL#GENERAL', 8: 'ROOMS#PRICES', 9: 'ROOM_AMENITIES#GENERAL', 10: 'ROOMS#COMFORT', 11: 'FACILITIES#GENERAL', 12: 'ROOM_AMENITIES#DESIGN&FEATURES', 13: 'HOTEL#DESIGN&FEATURES', 14: 'HOTEL#COMFORT', 15: 'HOTEL#QUALITY', 16: 'ROOM_AMENITIES#QUALITY', 17: 'HOTEL#PRICES', 18: 'HOTEL#CLEANLINESS', 19: 'ROOM_AMENITIES#COMFORT', 20: 'HOTEL#MISCELLANEOUS', 21: 'ROOMS#GENERAL', 22: 'ROOM_AMENITIES#MISCELLANEOUS', 23: 'ROOM_AMENITIES#CLEANLINESS', 24: 'FACILITIES#QUALITY', 25: 'FACILITIES#DESIGN&FEATURES', 26: 'FOOD&DRINKS#MISCELLANEOUS', 27: 'FACILITIES#CLEANLINESS', 28: 'FACILITIES#COMFORT', 29: 'FACILITIES#MISCELLANEOUS', 30: 'FOOD&DRINKS#PRICES', 31: 'FACILITIES#PRICES', 32: 'ROOMS#MISCELLANEOUS'}
}
polarity2id = {"positive": 0, "negative": 1, "neutral": 2}
id2polarity = {0: "positive", 1: "negative", 2: "neutral"}


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def pad_and_truncate(sequence, max_len=256, dtype="int64", eos_id=2):
    """Pad or truncate BERT input ids accordingly
    End of sentence [SEP] == 2
    """
    x = (np.ones(max_len) * 0).astype(dtype)  # init numpy array
    if len(sequence) > max_len:
        trunc = sequence[:max_len]
        trunc[-1] = eos_id
    else:
        trunc = sequence + [1] * (max_len - len(sequence))
    x[:len(trunc)] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_sequence_len, pretrained_bert_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name, use_fast=False)
        self.phobert = AutoModel.from_pretrained(pretrained_bert_name)
        self.max_sequence_len = max_sequence_len

    def text_to_sequence(self, text):
        tokens = self.tokenizer.tokenize(text)
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)

        return pad_and_truncate(input_id, self.max_sequence_len)


class ABSADataset:
    def __init__(self, fname, tokenizer):
        fin = open(fname, mode="r", encoding="utf-8", errors="ignore")
        lines = fin.readlines()
        fin.close()

        if "Hotel" in fname:
            aspect2id = aspect2ids["hotel"]
        else:
            aspect2id = aspect2ids["restaurant"]

        self.sentences = []
        self.aspects = []
        self.polarities = []
        self.sentence_input_ids = []
        self.polarity_input_ids = []
        self.aspect_input_ids = []
        self.attention_masks = []
        for i in tqdm(range(1, len(lines), 4), desc="Load and preprocess data: "):
            label_pattern = r"([\w#&]+)"  # keep joint aspects together ie. DESIGN&FEATURES
            labels = re.findall(label_pattern,
                                lines[i + 1].replace("\n", ""))  # [ASPECT', 'polarity',...] labels for single sentence

            aspect = []
            polarity = []
            for j in range(0, len(labels), 2):
                aspect.append(labels[j])
                polarity.append(labels[j + 1])

            sentence = lines[i].replace("\n", "").replace(".", "")
            word_segmented = word_tokenize(sentence, "text")  # PhoBert requires word_segmented text
            sentence_input_id = tokenizer.text_to_sequence("[CLS] " + word_segmented + " [SEP]")
            polarity_input_id = np.zeros(len(polarity2id))
            polarity_input_id[[polarity2id[x] for x in polarity]] = 1
            aspect_input_id = np.zeros(len(aspect2id))
            aspect_input_id[[aspect2id[x] for x in aspect]] = 1
            attention_mask = np.where(sentence_input_id == 1, 0, 1)

            self.sentences.append(sentence)
            self.aspects.append(aspect)
            self.polarities.append(polarity)
            self.sentence_input_ids.append(sentence_input_id)
            self.polarity_input_ids.append(polarity_input_id)
            self.aspect_input_ids.append(aspect_input_id)
            self.attention_masks.append(attention_mask)


    def __getitem__(self, index):
        datum = {
            "sentence": self.sentences[index],
            "aspects": self.aspects[index],
            "polarities": self.polarities[index],
            "sentence_input_ids": self.sentence_input_ids[index],
            "polarity_input_ids": self.polarity_input_ids[index],
            "aspect_input_ids": self.aspect_input_ids[index],
            "attention_masks": self.attention_masks[index],
        }
        return datum

    def __len__(self):
        return len(self.sentences)
