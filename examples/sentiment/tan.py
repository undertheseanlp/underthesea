import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from pytorch_lightning.loggers import WandbLogger


class MultiLabelClassificationDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        self.texts = [
            "Vừa qua tôi có dùng dịch vụ tại Khách Sạn TTC Hotel Premium Ngọc Lan Ngọc Lan Đà Lạt.",
            "Tuy nhiên buffet sáng ở đây không được ngon và chưa đa dạng lắm."
        ]
        self.labels = torch.FloatTensor([
            0,
            1
        ])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        return dict(
            text=text,
            label=self.labels[item],
            input_ids=self.tokenizer.encode(text, return_tensors='pt')
        )


class MultiLabelClassificationDatamodule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset)


class DEHClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(50257, 1)
        self.gpt2 = AutoModelWithLMHead.from_pretrained("imthanhlv/gpt2news")
        self.criterion = nn.MSELoss()

    def forward(self, input_ids, labels=None):
        loss = 0
        gpt2_outputs = self.gpt2(input_ids)
        hidden_states = gpt2_outputs[0]
        logits = self.linear(hidden_states)
        if labels is not None:
            loss = self.criterion(logits, labels)
            self.log('mse_loss', loss)
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        loss, outputs = self(input_ids, labels)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    model = DEHClassification()
    tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
    dataset = MultiLabelClassificationDataset(tokenizer=tokenizer)
    data = MultiLabelClassificationDatamodule(dataset)
    logger = WandbLogger(project='draft-sentiment-1', log_model=False)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=2,
        logger=logger)
    trainer.fit(model, data)
