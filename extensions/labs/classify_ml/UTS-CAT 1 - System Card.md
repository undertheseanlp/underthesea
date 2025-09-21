# UTS-CAT 1 - System Card: Vietnamese Text Classifier (VNTC)

## Model Details

### Model Description
- **Model Name**: UTS-CAT 1 (Underthesea Text Classification - Category 1)
- **Model Type**: TF-IDF + Logistic Regression Pipeline
- **Version**: 1.0
- **Created Date**: 2025-09-21
- **Framework**: scikit-learn >=1.6
- **Model File**: vntc_classifier.pkl
- **Training Data Hash**: 7421f92b

### Architecture
- **Feature Extraction**:
  - CountVectorizer with max_features=20,000 (optimized from 10,000)
  - N-gram range: (1, 2) - unigrams and bigrams
  - TF-IDF Transformer with IDF weighting
  - Caching enabled for vectorizer and TF-IDF components
- **Classifier**: Logistic Regression
  - Max iterations: 1,000
  - Random state: 42
  - Solver: Default (lbfgs)

## Intended Use

### Primary Use Cases
- Vietnamese news article classification
- Content categorization for Vietnamese text
- Document organization and tagging

### Supported Categories (10 classes)
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

## Training Data

### Dataset
- **Name**: VNTC (Vietnamese Text Classification) Dataset
- **Training Samples**: 33,759 documents
- **Test Samples**: 50,373 documents
- **Language**: Vietnamese
- **Format**: FastText format (__label__category followed by text)

### Data Distribution
- Balanced across 10 news categories
- Text preprocessing: None (raw Vietnamese text)
- Average document length: ~200-500 words

## Performance Metrics

### Overall Performance (Latest Results - 2025-09-21)
- **Training Accuracy**: 95.39% (improved from 94.93%)
- **Test Accuracy**: 92.33% (improved from 92.22%)
- **Training Time**: ~27.75 seconds (with caching: first run)
- **Training Time**: ~28.24 seconds (with caching: subsequent runs)
- **Inference Time**: ~19.26-20.33 seconds for 50,373 samples

### Per-Class Performance (Latest Run - Top 5 Classes)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|-----------|---------|
| chinh_tri_xa_hoi | 0.86 | 0.93 | 0.89 | 7,567 |
| doi_song | 0.81 | 0.71 | 0.76 | 2,036 |
| khoa_hoc | 0.88 | 0.79 | 0.83 | 2,096 |
| kinh_doanh | 0.94 | 0.88 | 0.91 | 5,276 |
| phap_luat | 0.92 | 0.92 | 0.92 | 3,788 |

### Aggregate Metrics (Latest Run)
- **Micro Average**: Precision: 0.89, Recall: 0.88, F1: 0.88
- **Macro Average**: Precision: 0.88, Recall: 0.85, F1: 0.86
- **Weighted Average**: Precision: 0.89, Recall: 0.88, F1: 0.88

### Performance Improvements
- **Feature Count**: Increased from 10,000 to 20,000 max features
- **Training Accuracy**: +0.46% improvement (94.93% → 95.39%)
- **Test Accuracy**: +0.11% improvement (92.22% → 92.33%)
- **Caching System**: Added hash-based caching for vectorizer and TF-IDF components

## Limitations

### Known Limitations
1. **Language Specificity**: Only works with Vietnamese text
2. **Domain Specificity**: Optimized for news articles, may not perform well on:
   - Social media posts
   - Technical documentation
   - Conversational text
3. **Feature Limitations**:
   - Limited to 10,000 most frequent features
   - May miss rare but important terms
4. **Class Confusion**: Lower performance on lifestyle (doi_song) category (71% recall)

### Biases
- Trained on news articles which may have formal writing style bias
- May reflect biases present in the original VNTC dataset
- Performance varies across categories (best on business/law, weakest on lifestyle)

## Usage

### Installation
```bash
pip install scikit-learn>=1.6 joblib
```

### Training
```bash
uv run --no-project --with 'scikit-learn>=1.6' python train.py
```

### Inference
```bash
# Single prediction
uv run --no-project --with 'scikit-learn>=1.6' python predict.py --text "Your Vietnamese text here"

# Interactive mode
uv run --no-project --with 'scikit-learn>=1.6' python predict.py --interactive

# Show examples
uv run --no-project --with 'scikit-learn>=1.6' python predict.py --examples
```

### Python API
```python
import joblib

# Load model
model = joblib.load('vntc_classifier.pkl')

# Make prediction
text = "Việt Nam giành chiến thắng trong trận bán kết"
prediction = model.predict([text])[0]
probabilities = model.predict_proba([text])[0]
```

## Model Files

### Required Files
1. **vntc_classifier.pkl** - Main model file (scikit-learn pipeline)
2. **label_mapping.txt** - List of category labels
3. **train.py** - Training script
4. **predict.py** - Inference script

### File Sizes (Approximate)
- Model file: ~30-50 MB (depending on vocabulary size)
- Label mapping: < 1 KB

## Ethical Considerations

### Recommended Use
- News categorization and organization
- Content management systems
- Research on Vietnamese text classification

### Not Recommended For
- Making decisions about individuals
- Content censorship
- Automated content moderation without human review

## Updates and Maintenance

### Version History
- **v1.0** (2025-09-21): Initial release with TF-IDF + Logistic Regression
  - Initial max_features: 10,000
  - Training accuracy: 94.93%, Test accuracy: 92.22%
- **v1.1** (2025-09-21): Performance optimization and caching
  - Increased max_features to 20,000
  - Added hash-based caching system for vectorizer and TF-IDF components
  - Training accuracy: 95.39%, Test accuracy: 92.33%
  - Added model selection options (SVC/Logistic Regression)
  - Implemented VNTCDataset class for better data handling

### Recent Improvements (v1.1)
1. **Performance Optimization**: Increased vocabulary size from 10k to 20k features
2. **Caching System**: Hash-based caching prevents recomputation of vectorization
3. **Model Selection**: Support for both SVC and Logistic Regression classifiers
4. **Better Architecture**: Modular design with Dataset class and command-line interface
5. **Cache Invalidation**: Automatic cache updates when training data changes

### Future Improvements
1. Experiment with more advanced models (XGBoost, Neural Networks)
2. Further increase vocabulary size for better coverage
3. Add support for longer documents
4. Implement confidence thresholds for uncertain predictions
5. Fine-tune on domain-specific data if needed

## Citation

If you use this model, please cite:
```
UTS-CAT 1 - Vietnamese Text Classifier v1.1
Based on VNTC Dataset
Trained using scikit-learn TF-IDF + Logistic Regression
Features: 20k max_features, (1,2) n-grams, hash-based caching
Training Data Hash: 7421f92b
Test Accuracy: 92.33%
2025
```

## License
Model trained on publicly available VNTC dataset. Please refer to original dataset license for usage terms.

## Contact
For questions or issues, please refer to the project repository.