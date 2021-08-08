import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


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
