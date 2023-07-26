# Word Tokenization

* [Google Colab](https://colab.research.google.com/drive/1NR9NlHJDj5_wywRze7yQizw1DgICKL72?usp=sharing)
* [Technical Report](technical_report.md)

## Usage

### Training

Train and Evaluate Model:

```
python train.py ++'dataset.include_test=False'
```

Train the Final Model (Including Test Split):

```
python train.py ++'dataset.include_test=True'
```

### Inference

Generate labels with the trained model

```
python predict.py +'output_dir="tmp/ws_20230725/"' +'text="Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"'
```
