<h1 align="center">Sonar Core 1 - System Card</h1>

<p align="center"><b>Underthesea Team</b></p>

<p align="center"><b>September 2025</b></p>

# Changelog

**2025-09-21**

- Initial release of Sonar Core 1

# Abstract

**Sonar Core 1** is a machine learning-based text classification model designed for Vietnamese language processing. Built on a **TF-IDF** (Term Frequency-Inverse Document Frequency) feature extraction pipeline combined with **Logistic Regression**, this model achieves **92.33% accuracy** on the VNTC (Vietnamese Text Classification) dataset across **10 news categories**. The model is specifically designed for Vietnamese news article classification, content categorization for Vietnamese text, and document organization and tagging. Developed as a base model to provide quick and reliable text classification support for **scikit-learn >=1.6** integration since **underthesea 8.1.0**, it employs optimized feature engineering with **20,000 max features** and bigram support, along with a hash-based caching system for efficient processing. This system card provides comprehensive documentation of the model's architecture, performance metrics, intended uses, and limitations.

# 1. Model Details

**Sonar Core 1** is a Vietnamese text classification model built on **scikit-learn >=1.6**, utilizing a TF-IDF pipeline with Logistic Regression to classify text across 10 news categories. The architecture employs:
- CountVectorizer with **20,000 max features** (optimized from the initial 10,000)
- N-gram extraction: unigram and bigram support
- TF-IDF transformation with IDF weighting
- Logistic Regression classifier with 1,000 max iterations
- **Hash-based caching system** for efficient processing

Released on **2025-09-21**, the model achieves **92.33% test accuracy** and **95.39% training accuracy** with optimized training time of approximately **28 seconds** using the hash-based caching system. The model features a dedicated VNTCDataset class for efficient data handling and improved modular architecture.

# 2. Training Data

## 2.1 Supported Categories (10 classes)
1. **chinh_tri_xa_hoi** - Politics and Society
2. **doi_song** - Lifestyle
3. **khoa_hoc** - Science
4. **kinh_doanh** - Business
5. **phap_luat** - Law
6. **suc_khoe** - Health
7. **the_gioi** - World News
8. **the_thao** - Sports
9. **van_hoa** - Culture
10. **vi_tinh** - Information Technology

## 2.2 Dataset
- **Name**: VNTC (Vietnamese Text Classification) Dataset
- **Training Samples**: 33,759 documents
- **Test Samples**: 50,373 documents
- **Language**: Vietnamese
- **Format**: FastText format (__label__category followed by text)

## 2.3 Data Distribution
- Balanced across 10 news categories
- Text preprocessing: None (raw Vietnamese text)
- Average document length: ~200-500 words

# 3. Performance Metrics

## 3.1 Overall Performance (Latest Results - 2025-09-21)
- **Training Accuracy**: 95.39% (improved from 94.93%)
- **Test Accuracy**: 92.33% (improved from 92.22%)
- **Training Time**: ~27.75 seconds (with caching: first run)
- **Training Time**: ~28.24 seconds (with caching: subsequent runs)
- **Inference Time**: ~19.26-20.33 seconds for 50,373 samples

## 3.2 Per-Class Performance (Latest Run - Top 5 Classes)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|-----------|---------|
| chinh_tri_xa_hoi | 0.86 | 0.93 | 0.89 | 7,567 |
| doi_song | 0.81 | 0.71 | 0.76 | 2,036 |
| khoa_hoc | 0.88 | 0.79 | 0.83 | 2,096 |
| kinh_doanh | 0.94 | 0.88 | 0.91 | 5,276 |
| phap_luat | 0.92 | 0.92 | 0.92 | 3,788 |

## 3.3 Aggregate Metrics (Latest Run)
- **Micro Average**: Precision: 0.89, Recall: 0.88, F1: 0.88
- **Macro Average**: Precision: 0.88, Recall: 0.85, F1: 0.86
- **Weighted Average**: Precision: 0.89, Recall: 0.88, F1: 0.88

## 3.4 Performance Improvements
- **Feature Count**: Increased from 10,000 to 20,000 max features
- **Training Accuracy**: +0.46% improvement (94.93% → 95.39%)
- **Test Accuracy**: +0.11% improvement (92.22% → 92.33%)
- **Caching System**: Added hash-based caching for vectorizer and TF-IDF components

# 4. Limitations

## 4.1 Known Limitations
1. **Language Specificity**: Only works with Vietnamese text
2. **Domain Specificity**: Optimized for news articles, may not perform well on:
   - Social media posts
   - Technical documentation
   - Conversational text
3. **Feature Limitations**:
   - Limited to 10,000 most frequent features
   - May miss rare but important terms
4. **Class Confusion**: Lower performance on lifestyle (doi_song) category (71% recall)

## 4.2 Biases
- Trained on news articles which may have formal writing style bias
- May reflect biases present in the original VNTC dataset
- Performance varies across categories (best on business/law, weakest on lifestyle)

# 5. Future Improvements

1. Experiment with more advanced models (XGBoost, Neural Networks)
2. Further increase vocabulary size for better coverage
3. Add support for longer documents
4. Implement confidence thresholds for uncertain predictions
5. Fine-tune on domain-specific data if needed

# 6. Usage

## 6.1 Installation
```bash
pip install scikit-learn>=1.6 joblib
```

## 6.2 Training
```bash
uv run --no-project --with 'scikit-learn>=1.6' python train.py
```

## 6.3 Inference
```bash
# Single prediction
uv run --no-project --with 'scikit-learn>=1.6' python predict.py --text "Your Vietnamese text here"

# Interactive mode
uv run --no-project --with 'scikit-learn>=1.6' python predict.py --interactive

# Show examples
uv run --no-project --with 'scikit-learn>=1.6' python predict.py --examples
```

## 6.4 Python API
```python
import joblib

# Load model
model = joblib.load('vntc_classifier.pkl')

# Make prediction
text = "Việt Nam giành chiến thắng trong trận bán kết"
prediction = model.predict([text])[0]
probabilities = model.predict_proba([text])[0]
```

# References

1. **VNTC Dataset**: Cong Duy Vu Hoang, Dien Dinh, Le Nguyen Nguyen, Quoc Hung Ngo. (2007). A Comparative Study on Vietnamese Text Classification Methods. In Proceedings of IEEE International Conference on Research, Innovation and Vision for the Future (RIVF 2007).

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Salton, G. and McGill, M.J. (1983). Introduction to Modern Information Retrieval. McGraw-Hill.

3. **Logistic Regression for Text Classification**: Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical Learning. Springer.

4. **Scikit-learn**: Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

5. **N-gram Language Models**: Brown, P.F., et al. (1992). Class-Based n-gram Models of Natural Language. Computational Linguistics, 18(4), 467-479.

# License
Model trained on publicly available VNTC dataset. Please refer to original dataset license for usage terms.

# Citation

If you use this model, please cite:

```bibtex
@techreport{underthesea2025sonarcore1,
  title = {Sonar Core 1: A Vietnamese Text Classification Model using Machine Learning},
  author = {Vu Anh},
  year = {2025},
  month = {September},
  institution = {Underthesea},
  version = {1.0},
  url = {https://github.com/undertheseanlp/underthesea/},
  keywords = {text classification, vietnamese nlp, machine learning, tf-idf, logistic regression}
}
```