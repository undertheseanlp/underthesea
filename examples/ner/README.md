# Vietnamese Named Entity Recognition

* [Colab Notebook](https://colab.research.google.com/drive/1bihzLZ7padXCOp0K9gCLvGzj-PGAJSKV)
* [Technical Report](https://www.overleaf.com/project/636b1deff54b3e38a674609b/detached)

Performance

* F1: 92.4

## Usage

Train model

```
python train_crf_local.py ++'dataset.include_test=True'
```

Use trained model to tag named entities

```
python predict_crf.py +'output_dir="tmp/ner_20220303/"' +'text="Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"'
```