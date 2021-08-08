import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchmetrics import F1
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from pytorch_lightning.loggers import WandbLogger
from underthesea.datasets.uit_absa_hotel.uit_absa_hotel import UITABSAHotel


class MultiLabelClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, num_labels, max_token_length: int = 50, samples=None):
        super().__init__()
        self.texts = [item["text"] for item in data]  # text
        self.labels = [item["aspect_label_ids"] for item in data]  # label ids
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_token_length = max_token_length
        self.samples = samples

    def __len__(self):
        length = len(self.texts)
        if self.samples is not None:
            length = min(self.samples, length)
        return length

    def __getitem__(self, item):
        text = self.texts[item]

        encoding = self.tokenizer(text, return_tensors='pt', padding="max_length", max_length=self.max_token_length)
        input_ids = encoding["input_ids"]
        hot_encoding = []
        for i in range(self.num_labels):
            if i in self.labels[item]:
                hot_encoding.append(1.0)
            else:
                hot_encoding.append(0.0)
        label_hot_encoding = torch.FloatTensor(hot_encoding)
        output = dict(
            text=text,
            label=label_hot_encoding,
            input_ids=input_ids
        )
        return output


class MultiLabelClassificationDatamodule(pl.LightningDataModule):
    def __init__(self, corpus, tokenizer, **kwargs):
        super().__init__()
        self.corpus = corpus
        # num_labels = corpus.num_labels
        num_labels = corpus.num_aspect_labels
        samples = None
        if "samples" in kwargs:
            samples = kwargs["samples"]
        del kwargs["samples"]
        self.dataset_kwargs = {
            "tokenizer": tokenizer,
            "num_labels": num_labels,
            "max_token_length": 300,
            "samples": samples
        }
        self.kwargs = kwargs

    def train_dataloader(self):
        dataset = MultiLabelClassificationDataset(self.corpus.train, **self.dataset_kwargs)
        return DataLoader(dataset, **self.kwargs)

    def val_dataloader(self):
        dataset = MultiLabelClassificationDataset(self.corpus.dev, **self.dataset_kwargs)
        return DataLoader(dataset, **self.kwargs)

    def test_dataloader(self):
        dataset = MultiLabelClassificationDataset(self.corpus.test, **self.dataset_kwargs)
        return DataLoader(dataset, **self.kwargs)


class GPT2TextClassification(pl.LightningModule):
    def __init__(self, gpt2, num_labels):
        super().__init__()
        self.gpt2 = gpt2
        vocab_size = self.gpt2.vocab_size
        self.linear = nn.Linear(vocab_size, num_labels)
        self.logit = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.train_f1 = F1(mdmc_average='global')
        self.valid_f1 = F1(mdmc_average='global')

    def forward(self, input_ids, labels=None):
        loss = 0
        gpt2_outputs = self.gpt2(input_ids)
        hidden_states = gpt2_outputs[0].squeeze()
        logits = self.logit(self.linear(hidden_states))
        batch_size, sequence_length = input_ids.shape[:2]
        logits = logits[range(batch_size), sequence_length]
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        loss, outputs = self(input_ids, labels)
        self.train_f1(outputs, labels.int())
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)
        self.log('train_loss', loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        loss, outputs = self(input_ids, labels)
        self.log('test_loss', loss)
        self.valid_f1(outputs, labels.int())
        self.log('valid_f1', self.valid_f1, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-6)
        return optimizer


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    print(config)
    tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    gpt2 = AutoModelWithLMHead.from_pretrained("imthanhlv/gpt2news")
    gpt2.resize_token_embeddings(len(tokenizer))
    gpt2.config.pad_token_id = gpt2.config.eos_token_id

    corpus = UITABSAHotel()
    # num_labels = corpus.num_labels
    num_labels = corpus.num_aspect_labels
    model = GPT2TextClassification(gpt2, num_labels)
    datamodule = MultiLabelClassificationDatamodule(corpus=corpus, tokenizer=tokenizer, **config.data)
    logger = WandbLogger(**config.logger)
    trainer = pl.Trainer(logger=logger, **config.trainer)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
