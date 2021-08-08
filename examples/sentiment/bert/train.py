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
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc

from tqdm import tqdm
from underthesea import word_tokenize
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertConfig, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


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
            # aspect_labels=binarizer(aspects, self.aspect2id),
            labels=binarizer(aspects, self.aspect2id),
            polarity_labels=binarizer(polarities, self.polarity2id),
        )


class UITABSADataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, test_dataset, batch_size=24):
        super().__init__()
        self.train_dataset  = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False)


class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, config, training_steps=None, warmup_steps=None):
    # def __init__(self, training_steps=None, warmup_steps=None):
        super().__init__()
        self.bert = BertForSequenceClassification(config)
        # self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)
        self.logit = nn.Sigmoid()
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.criterion = nn.BCELoss()
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask, labels=labels)
        pooled_output = output.logits
        pooled_output = self.dropout(pooled_output)
        logits = self.logit(self.classifier(pooled_output))
        output = self.classifier(output.logits)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        for i, name in enumerate(self.bert.config.num_labels):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer


def main():
    entity = "hotel"
    epochs = 5
    batch_size = 24
    num_labels = len(aspect2ids[entity])

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    train_dataset = UITABSADataset(fname=fnames[entity]["dev"], max_sequence_len=100, tokenizer=tokenizer)
    test_dataset = UITABSADataset(fname=fnames[entity]["dev"], max_sequence_len=100, tokenizer=tokenizer)

    # Config
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        architectures=['BertForSequenceClassification'],
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        num_labels=num_labels,
        pad_token_id=1,
        vocab_size= tokenizer.vocab_size
    )
    model = BertForMultilabelClassification(config, batch_size)
    # model.resize_token_embeddings(len(tokenizer))
    data_module = UITABSADataModule(train_dataset, test_dataset, batch_size)

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=5,
        progress_bar_refresh_rate=30
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()




