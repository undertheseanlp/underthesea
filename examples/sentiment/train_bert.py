import re
from os.path import join

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import F1

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, RobertaModel, BertPreTrainedModel, RobertaConfig
from underthesea import word_tokenize

DATASET_FOLDER = "/Users/taidnguyen/Desktop/Sentence-level-Hotel"
BERT_MODEL_NAME = "vinai/phobert-base"
fnames = dict(
    restaurant=dict(
        train=join(DATASET_FOLDER, "Train.txt"),
        test=join(DATASET_FOLDER, "Test.txt"),
        dev=join(DATASET_FOLDER, "Dev.txt")
    ),
    hotel=dict(
        train=join(DATASET_FOLDER, "Train_Hotel.txt"),
        test=join(DATASET_FOLDER, "Test_Hotel.txt"),
        dev=join(DATASET_FOLDER, "Dev_Hotel.txt")
    )
)

aspect2ids = {
    'restaurant': {'SERVICE#GENERAL': 0, 'RESTAURANT#PRICES': 1, 'RESTAURANT#GENERAL': 2, 'FOOD#STYLE&OPTIONS': 3,
                   'FOOD#QUALITY': 4, 'AMBIENCE#GENERAL': 5, 'RESTAURANT#MISCELLANEOUS': 6, 'LOCATION#GENERAL': 7,
                   'DRINKS#QUALITY': 8, 'FOOD#PRICES': 9, 'DRINKS#PRICES': 10, 'DRINKS#STYLE&OPTIONS': 11,
                   'OTHER_ASPECTS': 12},
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
                   8: 'DRINKS#QUALITY', 9: 'FOOD#PRICES', 10: 'DRINKS#PRICES', 11: 'DRINKS#STYLE&OPTIONS',
                   12: 'OTHER_ASPECTS'},
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
        # return 100
        return len(self.data)

    def __getitem__(self, index: int):
        data_indexed = self.data[index]
        text = data_indexed["sentence"]
        word_segmented = data_indexed["word_segmented"]  # PhoBert requires word_segmented text
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

        output = dict(
            text=text,
            word_segmented=word_segmented,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            # aspect_labels=binarizer(aspects, self.aspect2id),
            labels=binarizer(aspects, self.aspect2id),
            polarity_labels=binarizer(polarities, self.polarity2id),
        )
        return output


class UITABSADataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, test_dataset, val_dataset, batch_size=24):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # self.batch_size = batch_size
        self.batch_size = 6

    def train_dataloader(self):
        output = DataLoader(self.train_dataset, self.batch_size, shuffle=True, drop_last=True)
        return output

    def val_dataloader(self):
        output = DataLoader(self.val_dataset, self.batch_size, shuffle=True, drop_last=True)
        return output

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, drop_last=True)


class RobertaForABSA(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForABSA, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               head_mask=head_mask)
        return outputs


class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.roberta = RobertaForABSA.from_pretrained(BERT_MODEL_NAME)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        self.train_f1 = F1(mdmc_average='global')
        self.val_f1 = F1(mdmc_average='global')
        self.test_f1 = F1(mdmc_average='global')

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.roberta(input_ids.squeeze())
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.train_f1(outputs, labels.int())
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.val_f1(outputs, labels.int())
        self.log('val_f1', self.val_f1, on_step=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.test_f1(outputs, labels.int())
        self.log('test_f1', self.test_f1, on_step=True, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer


def main():
    entity = "hotel"
    # epochs = 5
    batch_size = 24
    n_classes = len(aspect2ids[entity])

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    train_dataset = UITABSADataset(fname=fnames[entity]["train"], max_sequence_len=100, tokenizer=tokenizer)
    val_dataset = UITABSADataset(fname=fnames[entity]["dev"], max_sequence_len=100, tokenizer=tokenizer)
    test_dataset = UITABSADataset(fname=fnames[entity]["test"], max_sequence_len=100, tokenizer=tokenizer)

    n_warmup_steps = 20
    n_training_steps = 100

    model = BertForMultilabelClassification(
        n_classes=n_classes,
        n_warmup_steps=n_warmup_steps,
        n_training_steps=n_training_steps
    )

    data_module = UITABSADataModule(train_dataset, val_dataset, test_dataset, batch_size)

    logger = WandbLogger(project='debug-phobert-sentiment')
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=5,
        progress_bar_refresh_rate=30,
        logger=logger
    )
    trainer.fit(model, data_module)
    trainer.test()


if __name__ == '__main__':
    main()
