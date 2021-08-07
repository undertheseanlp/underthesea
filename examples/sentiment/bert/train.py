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
import pytorch_lightning as pl
from tqdm import tqdm
from underthesea import word_tokenize
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

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

aspect2ids = {
    'restaurant': {'SERVICE#GENERAL': 0, 'RESTAURANT#PRICES': 1, 'RESTAURANT#GENERAL': 2, 'FOOD#STYLE&OPTIONS': 3,
                   'FOOD#QUALITY': 4, 'AMBIENCE#GENERAL': 5, 'RESTAURANT#MISCELLANEOUS': 6, 'LOCATION#GENERAL': 7,
                   'DRINKS#QUALITY': 8, 'FOOD#PRICES': 9, 'DRINKS#PRICES': 10, 'DRINKS#STYLE&OPTIONS': 11, 'OTHER_ASPECTS': 12},
    'hotel': {'ROOMS#DESIGN&FEATURES': 0, 'ROOMS#CLEANLINESS': 1, 'FOOD&DRINKS#STYLE&OPTIONS': 2, 'LOCATION#GENERAL': 3,
              'SERVICE#GENERAL': 4, 'ROOMS#QUALITY': 5, 'FOOD&DRINKS#QUALITY': 6, 'HOTEL#GENERAL': 7, 'ROOMS#PRICES': 8,
              'ROOM_AMENITIES#GENERAL': 9, 'ROOMS#COMFORT': 10, 'FACILITIES#GENERAL': 11,
              'ROOM_AMENITIES#DESIGN&FEATURES': 12, 'HOTEL#DESIGN&FEATURES': 13, 'HOTEL#COMFORT': 14,
              'HOTEL#QUALITY': 15, 'ROOM_AMENITIES#QUALITY': 16, 'HOTEL#PRICES': 17, 'HOTEL#CLEANLINESS': 18,
              'ROOM_AMENITIES#COMFORT': 19, 'HOTEL#MISCELLANEOUS': 20, 'ROOMS#GENERAL': 21,
              'ROOM_AMENITIES#MISCELLANEOUS': 22, 'ROOM_AMENITIES#CLEANLINESS': 23, 'FACILITIES#QUALITY': 24,
              'FACILITIES#DESIGN&FEATURES': 25, 'FOOD&DRINKS#MISCELLANEOUS': 26, 'FACILITIES#CLEANLINESS': 27,
              'FACILITIES#COMFORT': 28, 'FACILITIES#MISCELLANEOUS': 29, 'FOOD&DRINKS#PRICES': 30,
              'FACILITIES#PRICES': 31, 'ROOMS#MISCELLANEOUS': 32, 'OTHER_ASPECTS': 33}
}
id2aspects = {
    'restaurant': {0: 'SERVICE#GENERAL', 1: 'RESTAURANT#PRICES', 2: 'RESTAURANT#GENERAL', 3: 'FOOD#STYLE&OPTIONS',
                   4: 'FOOD#QUALITY', 5: 'AMBIENCE#GENERAL', 6: 'RESTAURANT#MISCELLANEOUS', 7: 'LOCATION#GENERAL',
                   8: 'DRINKS#QUALITY', 9: 'FOOD#PRICES', 10: 'DRINKS#PRICES', 11: 'DRINKS#STYLE&OPTIONS', 12: 'OTHER_ASPECTS'},
    'hotel': {0: 'ROOMS#DESIGN&FEATURES', 1: 'ROOMS#CLEANLINESS', 2: 'FOOD&DRINKS#STYLE&OPTIONS', 3: 'LOCATION#GENERAL',
              4: 'SERVICE#GENERAL', 5: 'ROOMS#QUALITY', 6: 'FOOD&DRINKS#QUALITY', 7: 'HOTEL#GENERAL', 8: 'ROOMS#PRICES',
              9: 'ROOM_AMENITIES#GENERAL', 10: 'ROOMS#COMFORT', 11: 'FACILITIES#GENERAL',
              12: 'ROOM_AMENITIES#DESIGN&FEATURES', 13: 'HOTEL#DESIGN&FEATURES', 14: 'HOTEL#COMFORT',
              15: 'HOTEL#QUALITY', 16: 'ROOM_AMENITIES#QUALITY', 17: 'HOTEL#PRICES', 18: 'HOTEL#CLEANLINESS',
              19: 'ROOM_AMENITIES#COMFORT', 20: 'HOTEL#MISCELLANEOUS', 21: 'ROOMS#GENERAL',
              22: 'ROOM_AMENITIES#MISCELLANEOUS', 23: 'ROOM_AMENITIES#CLEANLINESS', 24: 'FACILITIES#QUALITY',
              25: 'FACILITIES#DESIGN&FEATURES', 26: 'FOOD&DRINKS#MISCELLANEOUS', 27: 'FACILITIES#CLEANLINESS',
              28: 'FACILITIES#COMFORT', 29: 'FACILITIES#MISCELLANEOUS', 30: 'FOOD&DRINKS#PRICES',
              31: 'FACILITIES#PRICES', 32: 'ROOMS#MISCELLANEOUS', 33: 'OTHER_ASPECTS'}
}
polarity2id = {"positive": 0, "negative": 1, "neutral": 2}
id2polarity = {0: "positive", 1: "negative", 2: "neutral"}


def binarizer(labels: list, mapping: dict):
    binarized = torch.zeros(len(mapping))
    binarized[[mapping[x] for x in labels]] = 1
    return binarized


class UITABSADataset(Dataset):
    def __init__(self, fname: str, max_sequence_len: int, tokenizer: AutoTokenizer):
        fin = open(fname, mode="r", encoding="utf-8", errors="ignore")
        lines = fin.readlines()
        fin.close()

        self.polarity2id = polarity2id
        if "Hotel" in fname:
            self.aspect2id = aspect2ids["hotel"]
        else:
            self.aspect2id = aspect2ids["restaurant"]

        all_data = []
        for i in tqdm(range(1, len(lines), 4), desc="Load and preprocess data: "):
            label_pattern = r"([\w#&]+)"  # keep joint aspects together ie. DESIGN&FEATURES
            labels = re.findall(label_pattern,
                                lines[i + 1].replace("\n", ""))  # [ASPECT', 'polarity',...] labels for single sentence

            aspect = []
            polarity = []
            for j in range(0, len(labels), 2):
                asp = labels[j]
                if asp not in self.aspect2id.keys():
                    aspect.append("OTHER_ASPECTS")
                else:
                    aspect.append(labels[j])
                polarity.append(labels[j + 1])
            sentence = lines[i].replace("\n", "").replace(".", "")
            word_segmented = word_tokenize(sentence, "text")

            data = {
                "sentence": sentence,
                "word_segmented": word_segmented,
                "aspects": aspect,
                "polarities": polarity
            }
            all_data.append(data)

        self.data = all_data
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_indexed = self.data[index]
        text = data_indexed["sentence"]
        word_segmented = data_indexed["word_segmented"] # PhoBert requires word_segmented text
        aspects = data_indexed["aspects"]
        polarities = data_indexed["polarities"]
        encoded = self.tokenizer.encode_plus(
            text=word_segmented,
            add_special_tokens=True,
            max_length=self.max_sequence_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        return dict(
            text=text,
            word_segmented=word_segmented,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            aspect_labels=binarizer(aspects, self.aspect2id),
            polarity_labels=binarizer(polarities, self.polarity2id),
        )


class ToxicCommentDataModule(pl.LightningDataModule):

  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def setup(self, stage=None):
    self.train_dataset = ToxicCommentsDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )

    self.test_dataset = ToxicCommentsDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len

    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2
    )

  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )


def main():
    entity = "hotel"
    model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    train_dataset = UITABSADataset(fname=fnames[entity]["dev"], max_sequence_len=100, tokenizer=tokenizer)

if __name__ == '__main__':
    main()




