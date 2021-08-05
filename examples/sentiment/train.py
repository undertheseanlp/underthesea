import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from pytorch_lightning.loggers import WandbLogger

from underthesea.datasets.uit_absa_hotel.uit_absa_hotel import UITABSAHotel


class MultiLabelClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_length: int = 50):
        print("== init")
        super().__init__()
        self.texts = [item[1] for item in data]  # text
        labels = [item[3][0] for item in data]  # label ids
        self.labels = torch.FloatTensor(labels)
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        print("== getitem")
        text = self.texts[item]

        encoding = self.tokenizer(text, return_tensors='pt', padding="max_length", max_length=self.max_token_length)
        input_ids = encoding["input_ids"]
        output = dict(
            text=text,
            label=self.labels[item],
            input_ids=input_ids
        )
        return output


class MultiLabelClassificationDatamodule(pl.LightningDataModule):

    def __init__(self, corpus, tokenizer):
        print("MLCDatamodule pre init")
        super().__init__()
        print("MLCDatamodule post init")
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_token_length = 300
        self.batch_size = 1
        self.num_workers = 16
        print("MLCDatamodule fisnish init")

    def train_dataloader(self):
        print("==== train_dataloader")
        dataset = MultiLabelClassificationDataset(corpus.train, tokenizer, max_token_length=self.max_token_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = MultiLabelClassificationDataset(corpus.dev, tokenizer, max_token_length=self.max_token_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        dataset = MultiLabelClassificationDataset(corpus.test, tokenizer, max_token_length=self.max_token_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)


class GPT2TextClassification(pl.LightningModule):
    def __init__(self, gpt2):
        super().__init__()
        self.gpt2 = gpt2
        vocab_size = self.gpt2.vocab_size
        self.linear = nn.Linear(vocab_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, input_ids, labels=None):
        loss = 0
        gpt2_outputs = self.gpt2(input_ids)
        hidden_states = gpt2_outputs[0].squeeze()
        logits = self.linear(hidden_states)
        batch_size, sequence_length = input_ids.shape[:2]
        logits = logits[range(batch_size), sequence_length]
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        loss, outputs = self(input_ids, labels)
        self.log('train_loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        loss, outputs = self(input_ids, labels)
        self.log('test_loss', loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    print('====== main')
    tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    gpt2 = AutoModelWithLMHead.from_pretrained("imthanhlv/gpt2news")
    gpt2.resize_token_embeddings(len(tokenizer))
    gpt2.config.pad_token_id = gpt2.config.eos_token_id

    model = GPT2TextClassification(gpt2)
    corpus = UITABSAHotel()
    datamodule = MultiLabelClassificationDatamodule(corpus=corpus, tokenizer=tokenizer)
    print('===== log')
    logger = WandbLogger(project='draft-sentiment-2')
    print('===== Trainer')
    trainer = pl.Trainer(
        gpus=-1,
        accelerator='ddp',
        max_epochs=100,
        logger=logger)
    print('===== Fit')
    trainer.fit(model, datamodule=datamodule)
