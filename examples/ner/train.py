import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn


class VLSP2016NERDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()


class SequenceTokenClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 3)

    def dataloader(self):
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
    vlsp2016_ner = VLSP2016NERDataModule()
    trainer.fit(model=model, datamodule=vlsp2016_ner)

    result = trainer.test(test_dataloaders=test_loader)
