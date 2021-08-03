import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from pytorch_lightning.loggers import WandbLogger


class MultiLabelClassificationDataset(Dataset):
    def __init__(self, tokenizer, max_token_length: int = 50):
        super().__init__()
        self.texts = [
            "Tôi đi học",
            "tôi đi chơi",
            "tôi đi ăn",
            "Tuy nhiên buffet sáng ở đây không được ngon và chưa đa dạng lắm.",
            "Vừa qua tôi có dùng dịch vụ tại Khách Sạn TTC Hotel Premium Ngọc Lan Ngọc Lan Đà Lạt.",
            "Vừa qua tôi có dùng dịch vụ tại Khách Sạn TTC Hotel Premium Ngọc Lan Ngọc Lan Đà Lạt.",
        ]
        self.labels = torch.FloatTensor([
            0,
            0,
            0,
            1,
            1,
            1
        ])
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
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
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=2)


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
        self.log('mse_loss', loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    gpt2 = AutoModelWithLMHead.from_pretrained("imthanhlv/gpt2news")
    gpt2.resize_token_embeddings(len(tokenizer))
    gpt2.config.pad_token_id = gpt2.config.eos_token_id

    model = GPT2TextClassification(gpt2)
    dataset = MultiLabelClassificationDataset(tokenizer=tokenizer)
    data = MultiLabelClassificationDatamodule(dataset)
    logger = WandbLogger(project='draft-sentiment-2')
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        logger=logger)
    trainer.fit(model, data)
