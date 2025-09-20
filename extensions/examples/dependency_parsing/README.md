# Tutorial: Training a Vietnamese Dependency Parser

This part of the tutorial shows how you can train your own Vietnamese Dependency Parser using VLSP2020 Dependency Parsing dataset.

### Training a model

Install requirements

```
cd plays/dependency_parsing
conda create -n playground python=3.8
pip install -r requirements.txt
```

Train sample model

```
python vlsp2020_dp_sample.py
```

Train full model

```
python vlsp2020_dp_train.py
```
