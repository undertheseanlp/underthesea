Detail score after using MaltParser, we consider this result as baseline of our experiments  

*Table 2: detail score using MaltParser*
 
```
Metric     | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |    100.00 |    100.00 |    100.00 |    100.00
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
UFeats     |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |    100.00 |    100.00 |    100.00 |    100.00
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |     75.41 |     75.41 |     75.41 |     75.41
LAS        |     66.11 |     66.11 |     66.11 |     66.11
CLAS       |     62.70 |     62.17 |     62.43 |     62.17
MLAS       |     60.74 |     60.23 |     60.48 |     60.23
BLEX       |     62.70 |     62.17 |     62.43 |     62.17 
```

*To reproduce this result, you can run* 

```
export MALT_PARSER=/home/anhv/Downloads/maltparser-1.9.2
python malt_train.py 
```

UAS and LAS after training 240 epochs

```
2020-11-29 23:05:58,924 Epoch 240 saved
2020-11-29 23:05:58,924 dev:   - UCM: 30.67% LCM:  6.98% UAS: 87.28% LAS: 72.63%
2020-11-29 23:05:58,924 test:  - UCM: 30.67% LCM:  6.98% UAS: 87.28% LAS: 72.63%
2020-11-29 23:05:58,924 0:33:46.407770s elapsed, 0:00:05.960023s/epoch
```

*To reproduce this result, you can run* 

```
python nn_train.py 
```