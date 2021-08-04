import pytorch_lightning as pl
import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


class SimpleDataset(Dataset):
    def __init__(self):
        X = np.arange(10000)
        y = X * 2
        X = [[_] for _ in X]
        y = [[_] for _ in y]
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx]}


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.criterion = MSELoss()

    def forward(self, inputs_id, labels=None):
        outputs = self.fc(inputs_id)
        loss = 0
        if labels is not None:
            loss = self.criterion(outputs, labels)
        return loss, outputs

    def train_dataloader(self):
        dataset = SimpleDataset()
        return DataLoader(dataset, batch_size=1000)

    def training_step(self, batch, batch_idx):
        input_ids = batch["X"]
        labels = batch["y"]
        loss, outputs = self(input_ids, labels)
        self.log('mse_loss', loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    wandb_logger = WandbLogger(project='hugging-face')
    model = MyModel()
    trainer = pl.Trainer(max_epochs=600, gpus=1, logger=wandb_logger)
    trainer.fit(model)

    X = torch.Tensor([[1.0], [51.0], [89.0]])
    _, y = model(X)
    print(y)
