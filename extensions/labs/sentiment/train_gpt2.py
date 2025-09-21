import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import SGD
import torch.nn as nn
from torchmetrics import F1Score as F1
from transformers import AutoTokenizer, AutoModelWithLMHead
from pytorch_lightning.loggers import WandbLogger
from examples.sentiment.data import MultiLabelClassificationDatamodule
from underthesea.datasets.uit_absa_hotel.uit_absa_hotel import UITABSAHotel


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
