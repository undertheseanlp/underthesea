import torch
import hydra
import torch.nn as nn
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
from torchmetrics import F1Score as F1

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AdamW,
    RobertaModel,
    BertPreTrainedModel,
    RobertaConfig
)
from underthesea import word_tokenize
from underthesea.datasets.uit_absa_hotel.uit_absa_hotel import UITABSAHotel


class UITABSADataset(Dataset):
    def __init__(
        self, data, tokenizer: AutoTokenizer, num_labels, max_sequence_len: int = 100
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.num_labels = num_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_indexed = self.data[index]
        sentence = data_indexed["text"].replace("\n", "").replace(".", "")
        word_segmented = word_tokenize(
            sentence, "text"
        )  # PhoBert requires word_segmented text
        encoded = self.tokenizer.encode_plus(
            text=word_segmented,
            add_special_tokens=True,
            max_length=self.max_sequence_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        hot_encoding = []
        for i in range(self.num_labels):
            if i in data_indexed["label_ids"]:
                hot_encoding.append(1.0)
            else:
                hot_encoding.append(0.0)
        label_hot_encoding = torch.FloatTensor(hot_encoding)
        output = dict(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=label_hot_encoding
        )
        return output


class UITABSADataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, test_dataset, val_dataset, batch_size=24):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        output = DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, drop_last=True,
        )
        return output

    def val_dataloader(self):
        output = DataLoader(
            self.val_dataset, self.batch_size, shuffle=True, drop_last=True
        )
        return output

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, self.batch_size, shuffle=False, drop_last=True
        )


class RobertaForABSA(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForABSA, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        start_positions=None,
        end_positions=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask
        )
        return outputs


class BertForMultilabelClassification(pl.LightningModule):
    def __init__(
        self, bert_model, n_classes: int, n_training_steps=None, n_warmup_steps=None
    ):
        super().__init__()
        self.roberta = RobertaForABSA.from_pretrained(bert_model)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        self.train_f1 = F1(mdmc_average="global")
        self.val_f1 = F1(mdmc_average="global")
        self.test_f1 = F1(mdmc_average="global")

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
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.val_f1(outputs, labels.int())
        self.log("val_f1", self.val_f1, on_step=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.test_f1(outputs, labels.int())
        self.log("test_f1", self.test_f1, on_step=True, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    corpus = UITABSAHotel(training="aspect")  # predicting aspect or polarity
    batch_size = 24
    max_sequence_len = 100
    num_labels = corpus.num_labels
    BERT_MODEL_NAME = "vinai/phobert-base"

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=False)
    train_dataset = UITABSADataset(
        data=corpus.train,
        max_sequence_len=max_sequence_len,
        num_labels=num_labels,
        tokenizer=tokenizer,
    )
    val_dataset = UITABSADataset(
        data=corpus.dev,
        max_sequence_len=max_sequence_len,
        num_labels=num_labels,
        tokenizer=tokenizer
    )
    test_dataset = UITABSADataset(
        data=corpus.test,
        max_sequence_len=max_sequence_len,
        num_labels=num_labels,
        tokenizer=tokenizer
    )

    n_warmup_steps = 20
    n_training_steps = 100

    model = BertForMultilabelClassification(
        bert_model=BERT_MODEL_NAME,
        n_classes=num_labels,
        n_warmup_steps=n_warmup_steps,
        n_training_steps=n_training_steps
    )

    data_module = UITABSADataModule(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    # logger = WandbLogger(project="debug-phobert-sentiment")
    trainer = pl.Trainer(
        max_epochs=config.trainer.epoch,
        accelerator="cpu",
        enable_progress_bar=True,
        # logger=logger
    )
    trainer.fit(model, data_module)
    trainer.test()


if __name__ == "__main__":
    main()
