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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from underthesea import word_tokenize
from transformers import AutoTokenizer


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
        self.max_sequence_len = max_sequence_len

    def text_to_sequence(self, text):
        tokens = self.tokenizer.tokenize(text)
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)

        return pad_and_truncate(input_id, self.max_sequence_len)


polarity2id = {"positive": 0, "negative": 1, "neutral": 2}

class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, mode="r", encoding="utf-8", errors="ignore")
        lines = fin.readlines()
        fin.close()

        sentences = []
        aspects = []
        distinct_aspects = []
        polarities = []
        sentence_input_ids = []
        polarity_input_ids = []
        attention_masks = []
        for i in tqdm(range(1, len(lines), 4), desc="Load data: "):
            label_pattern = r"([\w#&]+)"  # keep joint aspects together ie. DESIGN&FEATURES
            labels = re.findall(label_pattern,
                                lines[i + 1].replace("\n", ""))  # [ASPECT', 'polarity',...] labels for single sentence

            sentence = lines[i].replace("\n", "").replace(".", "")
            aspect = []
            polarity = []
            for j in range(0, len(labels), 2):
                aspect.append(labels[j])
                polarity.append(labels[j + 1])
                if labels[j] not in distinct_aspects:
                    distinct_aspects.append(labels[j])

            word_segmented = word_tokenize(sentence, "text")  # PhoBert requires word_segmented text
            sentence_input_id = tokenizer.text_to_sequence("[CLS] " + word_segmented + " [SEP]")
            polarity_input_id = map(polarity2id.get, polarity)
            attention_mask = np.where(sentence_input_id == 1, 0, 1)

            sentences.append(sentence)
            aspects.append(aspect)
            polarities.append(polarity)
            sentence_input_ids.append(sentence_input_id)
            polarity_input_ids.append(polarity_input_id)
            attention_masks.append(attention_mask)

        all_data = {
            "sentences": sentences,
            "aspects": aspects,
            "polarities": polarities,
            "sentence_input_ids": sentence_input_ids,
            "polarity_input_ids": polarity_input_ids,
            "attention_masks": attention_masks
        }

        self.data = all_data
        self.distinct_aspects = distinct_aspects

    def __getitem__(self, index):
        datum = {
            "sentence": self.data["sentences"][index],
            "aspects": self.data["aspects"][index],
            "polarities": self.data["polarities"][index],
            "sentence_input_ids": self.data["sentence_input_ids"][index],
            "polarity_input_ids": self.data["polarity_input_ids"][index],
            "attention_masks": self.data["attention_masks"][index],

        }
        return datum

    def __len__(self):
        return len(self.data)


def main():
    tokenizer = Tokenizer4Bert(256, "vinai/phobert-base")
    ds = ABSADataset("/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Dev.txt", tokenizer)
    print(ds.distinct_aspects)


if __name__ == '__main__':
    main()
