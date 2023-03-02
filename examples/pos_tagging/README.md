# POS Taggin 

Train model 

```
python train_local.py ++'dataset.include_test=True'
```

Use trained model to get pos tag

```
python predict_crf.py +'output_dir="tmp/ner_20220303/"' +'text="Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"'
```