# Sentiment

## Train model with GPT-2

```
python train_gpt2.py data.batch_size=2 ++data.num_workers=16 logger.project=local-sentiment-5
```

Run with 100 samples

```
python train_gpt2.py data.batch_size=2 data.samples=100 logger.project=debug-sentiment-5
```

Run in cloud
```
python train_gpt2.py data.batch_size=16 ++data.num_workers=16 logger.project=cloud-sentiment-5
```

## Train model with Bert

```
python train_bert.py data.batch_size=2 ++data.num_workers=16 logger.project=local-sentiment-5
```

Run with 100 samples

```
python train_bert.py data.batch_size=2 data.samples=100 logger.project=debug-sentiment-5
```

Run in cloud
```
python train_bert.py data.batch_size=16 ++data.num_workers=16 logger.project=cloud-sentiment-5
```
