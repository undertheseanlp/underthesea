# Models

Underthesea provides pretrained models for various NLP tasks. This page describes the available models and how to use them.

## Model Management

### List Available Models

```bash
underthesea list-model
```

### Download a Model

```bash
underthesea download-model MODEL_NAME
```

### Model Storage Location

Models are stored in `~/.underthesea/models/`.

## Models by Task

### Word Segmentation

| Model | Description | Default |
|-------|-------------|---------|
| `WS_VLSP2013_CRF` | CRF model trained on VLSP 2013 | Yes |

### POS Tagging

| Model | Description | Default |
|-------|-------------|---------|
| `POS_VLSP2013_CRF` | CRF model trained on VLSP 2013 | Yes |

### Named Entity Recognition

| Model | Description | Default |
|-------|-------------|---------|
| `NER_VLSP2016_CRF` | CRF model trained on VLSP 2016 | Yes |
| `NER_VLSP2016_BERT` | BERT model trained on VLSP 2016 | `deep=True` |

Usage:

```python
from underthesea import ner

# CRF model (default)
ner("Việt Nam là quốc gia đông dân")

# BERT model (requires [deep] install)
ner("Việt Nam là quốc gia đông dân", deep=True)
```

### Text Classification

| Model | Description | Domain |
|-------|-------------|--------|
| `TC_VNTC_CRF` | News classification | Default |
| `TC_BANK_CRF` | Bank domain classification | `domain='bank'` |

Usage:

```python
from underthesea import classify

# General classification
classify("HLV đầu tiên ở Premier League bị sa thải")

# Bank domain
classify("Lãi suất từ BIDV rất ưu đãi", domain='bank')
```

### Sentiment Analysis

| Model | Description | Domain |
|-------|-------------|--------|
| `SA_GENERAL_CRF` | General sentiment | Default |
| `SA_BANK_CRF` | Bank domain sentiment | `domain='bank'` |

Usage:

```python
from underthesea import sentiment

# General sentiment
sentiment("Sản phẩm chất lượng tốt")

# Bank domain
sentiment("Xem lại vẫn thấy tự hào về BIDV", domain='bank')
```

### Translation

| Model | Description |
|-------|-------------|
| `TRANSLATION_VI_EN` | Vietnamese to English translation |

Usage:

```python
from underthesea import translate

translate("Hà Nội là thủ đô của Việt Nam")
# 'Hanoi is the capital of Vietnam'
```

### Text-to-Speech

| Model | Description | Architecture |
|-------|-------------|--------------|
| `VIET_TTS_V0_4_1` | Vietnamese TTS model | NAT + HiFi-GAN |

The TTS model uses a two-stage neural architecture:

1. **Duration + Acoustic Model**: Predicts phoneme durations and generates mel-spectrograms
2. **HiFi-GAN Vocoder**: Converts mel-spectrograms to high-quality audio

!!! note "Requires Installation"
    ```bash
    pip install "underthesea[voice]"
    ```

Download and usage:

```bash
underthesea download-model VIET_TTS_V0_4_1
```

```python
from underthesea.pipeline.say import say

say("Xin chào Việt Nam")
# Generates sound.wav
```

For detailed technical documentation, see [Voice Module Technical Report](../models/voice.md).

## Datasets

Underthesea also provides access to Vietnamese NLP datasets.

### List Available Datasets

```bash
underthesea list-data
```

### Download a Dataset

```bash
underthesea download-data DATASET_NAME
```

### Available Datasets

| Name | Type | Description |
|------|------|-------------|
| `VNTC` | Categorized | Vietnamese Text Classification corpus |
| `UTS2017-BANK` | Categorized | Bank domain classification |
| `UIT_ABSA_RESTAURANT` | Sentiment | Restaurant reviews |
| `UIT_ABSA_HOTEL` | Sentiment | Hotel reviews |
| `DI_Vietnamese-UVD` | Dictionary | Vietnamese dictionary |
| `CP_Vietnamese_VLC_v2_2022` | Plaintext | Vietnamese corpus |

### Dataset Storage Location

Datasets are stored in `~/.underthesea/datasets/`.

## Custom Models

You can also use custom models by specifying the model path:

```python
from underthesea import pos_tag

# Use a custom model
pos_tag("Xin chào", model="/path/to/custom/model")
```

## Model Performance

For benchmark results and performance comparisons, see the [GitHub repository](https://github.com/undertheseanlp/underthesea).
