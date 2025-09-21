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
- **Training Accuracy**: 95.39%
- **Test Accuracy**: 92.33%
- **Training Time**: ~27.18 seconds (with caching system)
- **Inference Time**: ~19.34 seconds for 50,373 samples

## 3.2 Per-Class Performance (Latest Run - All 10 Classes)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|-----------|---------|
| chinh_tri_xa_hoi | 0.86 | 0.93 | 0.89 | 7,567 |
| doi_song | 0.81 | 0.71 | 0.76 | 2,036 |
| khoa_hoc | 0.88 | 0.79 | 0.83 | 2,096 |
| kinh_doanh | 0.94 | 0.88 | 0.91 | 5,276 |
| phap_luat | 0.92 | 0.92 | 0.92 | 3,788 |
| suc_khoe | 0.93 | 0.95 | 0.94 | 5,417 |
| the_gioi | 0.95 | 0.93 | 0.94 | 6,716 |
| the_thao | 0.98 | 0.98 | 0.98 | 6,667 |
| van_hoa | 0.93 | 0.95 | 0.94 | 6,250 |
| vi_tinh | 0.94 | 0.95 | 0.94 | 4,560 |

## 3.3 Aggregate Metrics (Latest Run)
- **Overall Accuracy**: 92%
- **Macro Average**: Precision: 0.91, Recall: 0.90, F1: 0.91
- **Weighted Average**: Precision: 0.92, Recall: 0.92, F1: 0.92

## 3.4 Performance Analysis
- **Best Performing Categories**: Sports (the_thao) achieves 98% F1-score, followed by Health, World, Culture, and IT (all 94% F1-score)
- **Lowest Performing Category**: Lifestyle (doi_song) with 76% F1-score due to lower recall (71%)
- **Feature Count**: Uses 20,000 max features with bigram support
- **Caching System**: Hash-based caching for efficient vectorizer and TF-IDF processing

# 4. Limitations

## 4.1 Known Limitations
1. **Language Specificity**: Only works with Vietnamese text
2. **Domain Specificity**: Optimized for news articles, may not perform well on:
   - Social media posts
   - Technical documentation
   - Conversational text
3. **Feature Limitations**:
   - Limited to 20,000 most frequent features
   - May miss rare but important terms
4. **Class Confusion**: Lower performance on lifestyle (doi_song) category (71% recall)

## 4.2 Biases
- Trained on news articles which may have formal writing style bias
- May reflect biases present in the original VNTC dataset
- Performance varies across categories (best on sports at 98% F1-score, weakest on lifestyle at 76% F1-score)

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

1. VNTC Dataset: Hoang, Cong Duy Vu, Dien Dinh, Le Nguyen Nguyen, and Quoc Hung Ngo. (2007). A Comparative Study on Vietnamese Text Classification Methods. In Proceedings of IEEE International Conference on Research, Innovation and Vision for the Future (RIVF 2007), pp. 267-273. IEEE. DOI: 10.1109/RIVF.2007.369167

2. TF-IDF (Term Frequency-Inverse Document Frequency): Salton, Gerard, and Michael J. McGill. (1983). Introduction to Modern Information Retrieval. McGraw-Hill, New York. ISBN: 978-0070544840

3. Logistic Regression for Text Classification: Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). Springer Series in Statistics. Springer, New York. DOI: 10.1007/978-0-387-84858-7

4. Scikit-learn: Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12(85), 2825-2830. Retrieved from https://www.jmlr.org/papers/v12/pedregosa11a.html

5. N-gram Language Models: Brown, Peter F., Vincent J. Della Pietra, Peter V. deSouza, Jenifer C. Lai, and Robert L. Mercer. (1992). Class-Based n-gram Models of Natural Language. Computational Linguistics, 18(4), 467-480. Retrieved from https://aclanthology.org/J92-4003/

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