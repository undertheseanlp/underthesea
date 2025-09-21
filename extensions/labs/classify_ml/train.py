
import os
import urllib.request
import zipfile
import argparse
import pickle
import hashlib
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
import json
from abc import ABC, abstractmethod


class Dataset(ABC):
    """Base class for datasets"""

    @abstractmethod
    def load_data(self):
        """Load dataset and return X, y"""
        pass

    @abstractmethod
    def get_info(self):
        """Return dataset information"""
        pass


class VNTCDataset(Dataset):
    """VNTC Vietnamese Text Classification Dataset"""

    def __init__(self, dataset_folder=None):
        if dataset_folder is None:
            dataset_folder = os.path.expanduser("~/.underthesea/VNTC")

        self.dataset_folder = dataset_folder
        self.train_file = os.path.join(dataset_folder, "train.txt")
        self.test_file = os.path.join(dataset_folder, "test.txt")

        # Download dataset if not exists
        self._ensure_dataset_exists()

    def _ensure_dataset_exists(self):
        """Download dataset if not exists"""
        if not os.path.exists(self.train_file) or not os.path.exists(self.test_file):
            print("Dataset not found. Downloading VNTC dataset...")

            # Create directories
            os.makedirs(os.path.dirname(self.dataset_folder), exist_ok=True)

            # Download zip file
            zip_url = "https://github.com/undertheseanlp/underthesea/releases/download/resources/VNTC.zip"
            zip_path = os.path.join(os.path.dirname(self.dataset_folder), "VNTC.zip")

            print(f"Downloading from {zip_url}...")
            urllib.request.urlretrieve(zip_url, zip_path)

            # Extract zip file
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.dataset_folder))

            # Clean up zip file
            os.remove(zip_path)
            print("Dataset downloaded and extracted successfully!")

    def _parse_file(self, file_path):
        """Parse a single data file"""
        X_raw = []
        y = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    label = parts[0].replace('__label__', '')
                    text = parts[1]
                    y.append(label)
                    X_raw.append(text)
        return X_raw, y

    def load_data(self):
        """Load training and test data"""
        print("Loading VNTC dataset...")

        # Load training data
        print("Reading train.txt...")
        X_train_raw, y_train = self._parse_file(self.train_file)

        # Load test data
        print("Reading test.txt...")
        X_test_raw, y_test = self._parse_file(self.test_file)

        return (X_train_raw, y_train), (X_test_raw, y_test)

    def get_info(self):
        """Get dataset information"""
        (X_train_raw, y_train), (X_test_raw, y_test) = self.load_data()

        info = {
            'name': 'VNTC',
            'description': 'Vietnamese Text Classification Dataset',
            'train_samples': len(X_train_raw),
            'test_samples': len(X_test_raw),
            'unique_labels': len(set(y_train)),
            'labels': sorted(set(y_train))
        }

        return info


def get_available_models():
    """Get available classifier options"""
    return {
        'svc': ('SVC', SVC(kernel='linear', random_state=42, probability=True)),
        'logistic': ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42))
    }


def train_single_model(model_name, vect_max_features=20000, ngram_range=(1, 2)):
    """Train a single model with specified parameters"""
    # Initialize dataset
    dataset = VNTCDataset()
    output_folder = os.path.expanduser("~/.underthesea/models")

    # Load data using the VNTCDataset class
    (X_train_raw, y_train), (X_test_raw, y_test) = dataset.load_data()

    # Display dataset information
    info = dataset.get_info()
    print(f"Train samples: {info['train_samples']}")
    print(f"Test samples: {info['test_samples']}")
    print(f"Unique labels: {info['unique_labels']}")
    print(f"Labels: {info['labels'][:10]}...")  # Show first 10 labels

    # Get model
    available_models = get_available_models()
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list(available_models.keys())}")

    clf_name, classifier = available_models[model_name]

    # Configuration
    model_version = "UTS-C1"
    config_name = f"{model_version}_feat{vect_max_features//1000}k_ngram{ngram_range[0]}-{ngram_range[1]}_{clf_name}"

    print("\n" + "="*60)
    print(f"Training: {config_name}")
    print("="*60)

    # Create TF-IDF pipeline with caching
    print(f"Creating pipeline with max_features={vect_max_features}, ngram_range={ngram_range}, classifier={clf_name}")

    # Generate hash of training data for cache invalidation
    train_data_hash = hashlib.md5(''.join(X_train_raw).encode('utf-8')).hexdigest()[:8]
    print(f"Training data hash: {train_data_hash}")

    # Check for cached vectorizer and tfidf transformer
    cache_dir = os.path.join(output_folder, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    vect_cache_file = os.path.join(cache_dir, f'vectorizer_feat{vect_max_features}_ngram{ngram_range[0]}-{ngram_range[1]}_{train_data_hash}.pkl')
    tfidf_cache_file = os.path.join(cache_dir, f'tfidf_feat{vect_max_features}_ngram{ngram_range[0]}-{ngram_range[1]}_{train_data_hash}.pkl')

    # Try to load cached vectorizer and tfidf
    if os.path.exists(vect_cache_file) and os.path.exists(tfidf_cache_file):
        print("Loading cached vectorizer and TF-IDF transformer...")
        with open(vect_cache_file, 'rb') as f:
            vect = pickle.load(f)
        with open(tfidf_cache_file, 'rb') as f:
            tfidf = pickle.load(f)
        print("Cached components loaded successfully!")
    else:
        print("Creating and fitting new vectorizer and TF-IDF transformer...")
        # Create new components
        vect = CountVectorizer(max_features=vect_max_features, ngram_range=ngram_range)
        tfidf = TfidfTransformer(use_idf=True)

        # Fit vectorizer and tfidf on training data
        print("Fitting vectorizer on training data...")
        X_train_counts = vect.fit_transform(X_train_raw)
        print("Fitting TF-IDF transformer...")
        tfidf.fit(X_train_counts)

        # Cache the fitted components
        print("Caching vectorizer and TF-IDF transformer...")
        with open(vect_cache_file, 'wb') as f:
            pickle.dump(vect, f)
        with open(tfidf_cache_file, 'wb') as f:
            pickle.dump(tfidf, f)
        print("Components cached successfully!")

    text_clf = Pipeline([
        ('vect', vect),
        ('tfidf', tfidf),
        ('clf', classifier)
    ])

    # Train the model
    print("Training model...")
    start_time = time.time()
    text_clf.fit(X_train_raw, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    # Evaluate on training set
    print("Evaluating on training set...")
    train_predictions = text_clf.predict(X_train_raw)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set
    print("Evaluating on test set...")
    start_time = time.time()
    test_predictions = text_clf.predict(X_test_raw)
    test_accuracy = accuracy_score(y_test, test_predictions)
    prediction_time = time.time() - start_time
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.2f} seconds")

    # Show classification report for first 5 classes
    print("\nClassification Report (first 5 classes):")
    unique_labels = sorted(set(y_train))[:5]
    report = classification_report(y_test, test_predictions, labels=unique_labels, zero_division=0, output_dict=True)
    print(classification_report(y_test, test_predictions, labels=unique_labels, zero_division=0))

    # Save the model
    import joblib
    os.makedirs(output_folder, exist_ok=True)
    model_filename = os.path.join(output_folder, f'{config_name}.pkl')
    joblib.dump(text_clf, model_filename)
    print(f"Model saved to {model_filename}")

    # Save as main model
    main_model_path = os.path.join(output_folder, 'vntc_classifier.pkl')
    joblib.dump(text_clf, main_model_path)
    print(f"Model saved as main model to {main_model_path}")

    # Save label mapping
    label_mapping_filename = os.path.join(output_folder, 'label_mapping.txt')
    with open(label_mapping_filename, 'w', encoding='utf-8') as f:
        for label in sorted(set(y_train)):
            f.write(f"{label}\n")
    print(f"Label mapping saved to {label_mapping_filename}")

    return {
        'config_name': config_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'prediction_time': prediction_time
    }


def train_all_models():
    """Train all model combinations (original behavior)"""
    # Initialize dataset
    dataset = VNTCDataset()
    output_folder = os.path.expanduser("~/.underthesea/models")

    # Load data using the VNTCDataset class
    (X_train_raw, y_train), (X_test_raw, y_test) = dataset.load_data()

    # Display dataset information
    info = dataset.get_info()
    print(f"Train samples: {info['train_samples']}")
    print(f"Test samples: {info['test_samples']}")
    print(f"Unique labels: {info['unique_labels']}")
    print(f"Labels: {info['labels'][:10]}...")  # Show first 10 labels

    # Configuration options for experiments
    model_version = "UTS-C1"
    max_features_options = [10000, 20000, 30000]
    ngram_options = [(1, 2), (1, 3)]
    classifier_options = [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('SVC', SVC(kernel='linear', random_state=42, probability=True))
    ]

    # Store results for all experiments
    results = []

    # Run experiments with different configurations
    for max_features in max_features_options:
        for ngram_range in ngram_options:
            for clf_name, classifier in classifier_options:
                config_name = f"{model_version}_feat{max_features//1000}k_ngram{ngram_range[0]}-{ngram_range[1]}_{clf_name}"
                print("\n" + "="*60)
                print(f"Training: {config_name}")
                print("="*60)

                # Create TF-IDF pipeline
                print(f"Creating pipeline with max_features={max_features}, ngram_range={ngram_range}, classifier={clf_name}")
                text_clf = Pipeline([
                    ('vect', CountVectorizer(max_features=max_features, ngram_range=ngram_range)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('clf', classifier)
                ])

                # Train the model
                print("Training model...")
                start_time = time.time()
                text_clf.fit(X_train_raw, y_train)
                train_time = time.time() - start_time
                print(f"Training completed in {train_time:.2f} seconds")

                # Evaluate on training set
                print("Evaluating on training set...")
                train_predictions = text_clf.predict(X_train_raw)
                train_accuracy = accuracy_score(y_train, train_predictions)
                print(f"Training accuracy: {train_accuracy:.4f}")

                # Evaluate on test set
                print("Evaluating on test set...")
                start_time = time.time()
                test_predictions = text_clf.predict(X_test_raw)
                test_accuracy = accuracy_score(y_test, test_predictions)
                prediction_time = time.time() - start_time
                print(f"Test accuracy: {test_accuracy:.4f}")
                print(f"Prediction time: {prediction_time:.2f} seconds")

                # Store results
                result = {
                    'model_version': model_version,
                    'config_name': config_name,
                    'max_features': max_features,
                    'ngram_range': ngram_range,
                    'classifier': clf_name,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_time': train_time,
                    'prediction_time': prediction_time
                }
                results.append(result)

                # Show classification report for first 5 classes
                print("\nClassification Report (first 5 classes):")
                unique_labels = sorted(set(y_train))[:5]
                print(classification_report(y_test, test_predictions, labels=unique_labels, zero_division=0))

                # Save the model with configuration name
                import joblib
                os.makedirs(output_folder, exist_ok=True)
                model_filename = os.path.join(output_folder, f'{config_name}.pkl')
                joblib.dump(text_clf, model_filename)
                print(f"Model saved to {model_filename}")

    # Print summary of all experiments
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Config':<50} {'Train Acc':<10} {'Test Acc':<10} {'Train Time':<12} {'Pred Time':<10}")
    print("-"*80)
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
        print(f"{result['config_name']:<50} {result['train_accuracy']:.4f}     {result['test_accuracy']:.4f}      {result['train_time']:>8.2f}s    {result['prediction_time']:>6.2f}s")

    # Save best model as the main model
    best_result = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest configuration: {best_result['config_name']} with test accuracy: {best_result['test_accuracy']:.4f}")

    # Load and save best model as main model
    import joblib
    best_model_path = os.path.join(output_folder, f"{best_result['config_name']}.pkl")
    best_model = joblib.load(best_model_path)
    main_model_path = os.path.join(output_folder, 'vntc_classifier.pkl')
    joblib.dump(best_model, main_model_path)
    print(f"Best model saved as main model to {main_model_path}")

    # Save results to JSON
    results_file = os.path.join(output_folder, f'{model_version}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    # Save label mapping for reference
    label_mapping_filename = os.path.join(output_folder, 'label_mapping.txt')
    with open(label_mapping_filename, 'w', encoding='utf-8') as f:
        for label in sorted(set(y_train)):
            f.write(f"{label}\n")
    print(f"Label mapping saved to {label_mapping_filename}")

    return results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train VNTC text classification models')
    parser.add_argument('--model', type=str, choices=['svc', 'logistic'], default='svc',
                        help='Model to train (default: svc)')
    parser.add_argument('--max-features', type=int, default=20000,
                        help='Maximum number of features for TF-IDF (default: 20000)')
    parser.add_argument('--ngram-min', type=int, default=1,
                        help='Minimum n-gram range (default: 1)')
    parser.add_argument('--ngram-max', type=int, default=2,
                        help='Maximum n-gram range (default: 2)')
    parser.add_argument('--all-models', action='store_true',
                        help='Train all model combinations instead of single model')

    args = parser.parse_args()

    if args.all_models:
        print("Training all model combinations...")
        results = train_all_models()
    else:
        print(f"Training single model: {args.model}")
        ngram_range = (args.ngram_min, args.ngram_max)
        result = train_single_model(args.model, args.max_features, ngram_range)
        print(f"\nTraining completed: {result['config_name']}")
        print(f"Test accuracy: {result['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()