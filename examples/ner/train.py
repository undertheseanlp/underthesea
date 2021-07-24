import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn


class SequenceTokenClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 3)

    def train_dataloader(self) -> DataLoader:
        pass

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.backbone(X)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters())


if __name__ == '__main__':
    train_loader = None
    val_loader = None
    test_loader = None
    model = SequenceTokenClassifier()

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

    result = trainer.test(test_dataloaders=test_loader)
