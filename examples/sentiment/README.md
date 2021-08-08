# Sentiment

Run `train.py`

```
python train.py  
```

Run `train.py` with 100 samples 

```
python train.py data.batch_size=2 data.samples=100 logger.project=draft-sentiment-5-debug 
```

Run in clouod
```
python train.py data.batch_size=16 ++data.num_workers=16 logger.project=draft-sentiment-5
```

