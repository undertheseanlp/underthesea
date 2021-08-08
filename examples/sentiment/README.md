# Sentiment

Run `train.py`

```
python train.py data.batch_size=2 ++data.num_workers=16 logger.project=local-sentiment-5
```

Run `train.py` with 100 samples

```
python train.py data.batch_size=2 data.samples=100 logger.project=debug-sentiment-5 
```

Run in clouod
```
python train.py data.batch_size=16 ++data.num_workers=16 logger.project=cloud-sentiment-5
```

