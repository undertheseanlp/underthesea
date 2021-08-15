import logging
from os.path import join

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
import torch.nn as nn
import torch
from torchmetrics import F1
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

BERT_MODEL_NAME = "vinai/phobert-base"


class LabelEncoder:
    def __init__(self):
        self.index2label = {}
        self.label2index = {}
        self.vocab_size = 0

    def encode(self, labels):
        if type(labels) == list:
            return [self.encode(label) for label in labels]
        label = labels  # label is a string
        if label not in self.label2index:
            index = self.vocab_size
            self.label2index[label] = index
            self.index2label[index] = label
            self.vocab_size += 1
            return index
        else:
            return self.label2index[label]


class TokenClassificationCorpus:
    """ Load data from disk
    """

    def __init__(self, corpus_folder):
        train_file = join(corpus_folder, "train.txt")
        val_file = join(corpus_folder, "dev.txt")
        test_file = join(corpus_folder, "test.txt")
        self.label_encoder = LabelEncoder()
        self.train = self._extract_sentences(train_file)
        self.val = self._extract_sentences(val_file)
        self.test = self._extract_sentences(test_file)
        self.num_labels = self.label_encoder.vocab_size

    def _extract_sentence(self, sentence):
        rows = sentence.split("\n")
        rows = [_.split(" ") for _ in rows]
        tokens = [_[0] for _ in rows]
        labels = [_[1] for _ in rows]
        labels_encoded = self.label_encoder.encode(labels)
        return {
            "tokens": tokens,
            "labels": labels_encoded
        }

    def _extract_sentences(self, file):
        with open(file) as f:
            content = f.read().strip()
            sentences = content.split("\n\n")
            sentences = [self._extract_sentence(s) for s in sentences]
        return sentences


class TokenClassificationDataset(Dataset):
    """ Tokenize
    """

    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.max_sequence_len = 256
        self.tokenizer = tokenizer

    def __len__(self):
        # return min(10, len(self.data))
        return len(self.data)

    def _pad_labels(self, labels):
        if len(labels) >= self.max_sequence_len:
            return labels[self.max_sequence_len]

        return labels + [-100] * (self.max_sequence_len - len(labels))

    def __getitem__(self, item):
        value = self.data[item]
        tokens = value["tokens"]
        text = " ".join(tokens)
        encoded = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_sequence_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        encoded["labels"] = torch.LongTensor(self._pad_labels(value["labels"]))
        return encoded


class TokenClassificationDataModule(pl.LightningModule):
    """ Loader
    """

    def __init__(self, corpus, tokenizer):
        super().__init__()
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.batch_size = 30
        self.num_workers = 8

    def train_dataloader(self):
        dataset = TokenClassificationDataset(self.corpus.train, self.tokenizer)
        if self.num_workers:
            return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers, shuffle=True)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        dataset = TokenClassificationDataset(self.corpus.val, self.tokenizer)
        if self.num_workers:
            return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers, shuffle=True)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def test_dataloader(self):
        dataset = TokenClassificationDataset(self.corpus.test, self.tokenizer)
        if self.num_workers:
            return DataLoader(dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, drop_last=True)


class BertForTokenClassification(pl.LightningModule):
    # feed bert model
    def __init__(self, num_labels):
        super().__init__()
        self.hidden_size = 768
        self.num_labels = num_labels
        self.linear = nn.Linear(self.hidden_size, self.num_labels)
        self.train_loss = nn.CrossEntropyLoss()
        self.train_f1 = F1()
        self.validation_f1 = F1()

        self.transformer = AutoModel.from_pretrained(BERT_MODEL_NAME)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids.squeeze(), attention_mask=attention_mask.squeeze())
        sequence_output = outputs[0]
        logits = self.linear(sequence_output)
        loss = None
        if labels is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(self.train_loss.ignore_index).type_as(labels)
            )
            loss = self.train_loss(active_logits, active_labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('validation_loss', loss)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer


def main():
    corpus_folder = "bert-ner/data"
    corpus = TokenClassificationCorpus(corpus_folder)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=False)
    datamodule = TokenClassificationDataModule(corpus, tokenizer)
    model = BertForTokenClassification(num_labels=corpus.num_labels)
    logger = WandbLogger(project='phobert-token-classification', log_model=False)
    trainer = pl.Trainer(
        max_epochs=20,
        checkpoint_callback=False,
        gpus=-1,
        logger=logger
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
