import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from datasets import list_datasets, load_dataset


def setup_logging(run_name):
    """Setup logging to save all information to runs folder"""
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)

    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return run_dir




def get_available_models():
    """Get available classifier options"""
    return {
        'svc': ('SVC', SVC(kernel='linear', random_state=42, probability=True)),
        'logistic': ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42))
    }


def train_single_model(model_name, dataset_name='vntc', vect_max_features=20000, ngram_range=(1, 2), n_samples=None):
    """Train a single model with specified parameters

    Args:
        model_name: Name of the model to train ('svc' or 'logistic')
        dataset_name: Name of the dataset to use ('vntc' or 'uts2017_bank')
        vect_max_features: Maximum number of features for vectorizer
        ngram_range: N-gram range for feature extraction
        n_samples: Optional limit on number of samples to load
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = setup_logging(timestamp)

    logging.info(f"Starting training run: {timestamp}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Max features: {vect_max_features}")
    logging.info(f"N-gram range: {ngram_range}")
    if n_samples:
        logging.info(f"Sample limit: {n_samples}")

    # Initialize dataset
    dataset = load_dataset(dataset_name, n_samples=n_samples)
    output_folder = os.path.join(run_dir, "models")
    os.makedirs(output_folder, exist_ok=True)

    # Load data using the dataset
    logging.info("Loading dataset...")
    (X_train_raw, y_train), (X_test_raw, y_test) = dataset.load_data()

    # Display dataset information
    info = dataset.get_info()
    logging.info(f"Train samples: {info['train_samples']}")
    logging.info(f"Test samples: {info['test_samples']}")
    logging.info(f"Unique labels: {info['unique_labels']}")
    logging.info(f"Labels: {info['labels'][:10]}...")  # Show first 10 labels
    print(f"Train samples: {info['train_samples']}")
    print(f"Test samples: {info['test_samples']}")
    print(f"Unique labels: {info['unique_labels']}")
    print(f"Labels: {info['labels'][:10]}...")  # Show first 10 labels

    # Get model
    available_models = get_available_models()
    if model_name not in available_models:
        error_msg = f"Model '{model_name}' not available. Choose from: {list(available_models.keys())}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    clf_name, classifier = available_models[model_name]
    logging.info(f"Selected classifier: {clf_name}")

    # Configuration
    model_version = f"UTS-C1-{dataset_name.upper()}"
    config_name = f"{model_version}_feat{vect_max_features // 1000}k_ngram{ngram_range[0]}-{ngram_range[1]}_{clf_name}"

    logging.info("=" * 60)
    logging.info(f"Training: {config_name}")
    logging.info("=" * 60)
    print("\n" + "=" * 60)
    print(f"Training: {config_name}")
    print("=" * 60)

    # Create TF-IDF pipeline with caching
    logging.info(f"Creating pipeline with max_features={vect_max_features}, ngram_range={ngram_range}, classifier={clf_name}")
    print(f"Creating pipeline with max_features={vect_max_features}, ngram_range={ngram_range}, classifier={clf_name}")

    # Generate hash of training data for cache invalidation
    train_data_hash = hashlib.md5(''.join(X_train_raw).encode('utf-8')).hexdigest()[:8]
    logging.info(f"Training data hash: {train_data_hash}")
    print(f"Training data hash: {train_data_hash}")

    # Check for cached vectorizer and tfidf transformer
    cache_dir = os.path.expanduser("~/.underthesea/cache")
    os.makedirs(cache_dir, exist_ok=True)

    vect_cache_file = os.path.join(cache_dir, f'vectorizer_{dataset_name}_feat{vect_max_features}_ngram{ngram_range[0]}-{ngram_range[1]}_{train_data_hash}.pkl')
    tfidf_cache_file = os.path.join(cache_dir, f'tfidf_{dataset_name}_feat{vect_max_features}_ngram{ngram_range[0]}-{ngram_range[1]}_{train_data_hash}.pkl')

    # Try to load cached vectorizer and tfidf
    if os.path.exists(vect_cache_file) and os.path.exists(tfidf_cache_file):
        logging.info("Loading cached vectorizer and TF-IDF transformer...")
        print("Loading cached vectorizer and TF-IDF transformer...")
        with open(vect_cache_file, 'rb') as f:
            vect = pickle.load(f)
        with open(tfidf_cache_file, 'rb') as f:
            tfidf = pickle.load(f)
        logging.info("Cached components loaded successfully!")
        print("Cached components loaded successfully!")
    else:
        logging.info("Creating and fitting new vectorizer and TF-IDF transformer...")
        print("Creating and fitting new vectorizer and TF-IDF transformer...")
        # Create new components
        vect = CountVectorizer(max_features=vect_max_features, ngram_range=ngram_range)
        tfidf = TfidfTransformer(use_idf=True)

        # Fit vectorizer and tfidf on training data
        logging.info("Fitting vectorizer on training data...")
        print("Fitting vectorizer on training data...")
        X_train_counts = vect.fit_transform(X_train_raw)
        logging.info("Fitting TF-IDF transformer...")
        print("Fitting TF-IDF transformer...")
        tfidf.fit(X_train_counts)

        # Cache the fitted components
        logging.info("Caching vectorizer and TF-IDF transformer...")
        print("Caching vectorizer and TF-IDF transformer...")
        with open(vect_cache_file, 'wb') as f:
            pickle.dump(vect, f)
        with open(tfidf_cache_file, 'wb') as f:
            pickle.dump(tfidf, f)
        logging.info("Components cached successfully!")
        print("Components cached successfully!")

    text_clf = Pipeline([
        ('vect', vect),
        ('tfidf', tfidf),
        ('clf', classifier)
    ])

    # Train the model
    logging.info("Training model...")
    print("Training model...")
    start_time = time.time()
    text_clf.fit(X_train_raw, y_train)
    train_time = time.time() - start_time
    logging.info(f"Training completed in {train_time:.2f} seconds")
    print(f"Training completed in {train_time:.2f} seconds")

    # Evaluate on training set
    logging.info("Evaluating on training set...")
    print("Evaluating on training set...")
    train_predictions = text_clf.predict(X_train_raw)
    train_accuracy = accuracy_score(y_train, train_predictions)
    logging.info(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set
    logging.info("Evaluating on test set...")
    print("Evaluating on test set...")
    start_time = time.time()
    test_predictions = text_clf.predict(X_test_raw)
    test_accuracy = accuracy_score(y_test, test_predictions)
    prediction_time = time.time() - start_time
    logging.info(f"Test accuracy: {test_accuracy:.4f}")
    logging.info(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.2f} seconds")

    # Show classification report for all classes
    logging.info("Classification Report (all classes):")
    print("\nClassification Report (all classes):")
    report = classification_report(y_test, test_predictions, zero_division=0, output_dict=True)
    logging.info(classification_report(y_test, test_predictions, zero_division=0))
    print(classification_report(y_test, test_predictions, zero_division=0))

    # Save the model
    import joblib
    os.makedirs(output_folder, exist_ok=True)

    # Save as main model in run directory
    main_model_path = os.path.join(output_folder, 'model.pkl')
    joblib.dump(text_clf, main_model_path)
    logging.info(f"Model saved to {main_model_path}")
    print(f"Model saved to {main_model_path}")

    # Save label mapping
    label_mapping_filename = os.path.join(output_folder, 'labels.txt')
    with open(label_mapping_filename, 'w', encoding='utf-8') as f:
        for label in sorted(set(y_train)):
            f.write(f"{label}\n")
    logging.info(f"Label mapping saved to {label_mapping_filename}")
    print(f"Label mapping saved to {label_mapping_filename}")

    # Save dataset info
    dataset_info_file = os.path.join(output_folder, 'dataset_info.json')
    with open(dataset_info_file, 'w') as f:
        json.dump({
            'dataset_name': dataset_name,
            'n_samples': n_samples,
            'info': info
        }, f, indent=2)
    logging.info(f"Dataset info saved to {dataset_info_file}")

    # Save run metadata to runs folder
    run_metadata = {
        'timestamp': timestamp,
        'config_name': config_name,
        'dataset_name': dataset_name,
        'n_samples': n_samples,
        'model_name': model_name,
        'vect_max_features': vect_max_features,
        'ngram_range': ngram_range,
        'classifier': clf_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'prediction_time': prediction_time,
        'train_samples': info['train_samples'],
        'test_samples': info['test_samples'],
        'unique_labels': info['unique_labels'],
        'classification_report': report
    }

    metadata_file = os.path.join(run_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(run_metadata, f, indent=2)
    logging.info(f"Run metadata saved to {metadata_file}")

    logging.info(f"Training run completed: {timestamp}")

    return {
        'config_name': config_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'prediction_time': prediction_time,
        'run_dir': run_dir
    }


def train_all_models(dataset_name='vntc', n_samples=None):
    """Train all model combinations

    Args:
        dataset_name: Name of the dataset to use ('vntc' or 'uts2017_bank')
        n_samples: Optional limit on number of samples to load
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = setup_logging(timestamp)

    logging.info(f"Starting training run for all models: {timestamp}")
    logging.info(f"Dataset: {dataset_name}")
    if n_samples:
        logging.info(f"Sample limit: {n_samples}")

    # Initialize dataset
    dataset = load_dataset(dataset_name, n_samples=n_samples)
    output_folder = os.path.join(run_dir, "models")
    os.makedirs(output_folder, exist_ok=True)

    # Load data using the dataset
    logging.info("Loading dataset...")
    (X_train_raw, y_train), (X_test_raw, y_test) = dataset.load_data()

    # Display dataset information
    info = dataset.get_info()
    logging.info(f"Train samples: {info['train_samples']}")
    logging.info(f"Test samples: {info['test_samples']}")
    logging.info(f"Unique labels: {info['unique_labels']}")
    logging.info(f"Labels: {info['labels'][:10]}...")  # Show first 10 labels
    print(f"Train samples: {info['train_samples']}")
    print(f"Test samples: {info['test_samples']}")
    print(f"Unique labels: {info['unique_labels']}")
    print(f"Labels: {info['labels'][:10]}...")  # Show first 10 labels

    # Configuration options for experiments
    model_version = f"UTS-C1-{dataset_name.upper()}"
    max_features_options = [10000, 20000, 30000]
    ngram_options = [(1, 2), (1, 3)]
    classifier_options = [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('SVC', SVC(kernel='linear', random_state=42, probability=True))
    ]

    logging.info(f"Training configurations: {len(max_features_options) * len(ngram_options) * len(classifier_options)} total")

    # Store results for all experiments
    results = []

    # Run experiments with different configurations
    for max_features in max_features_options:
        for ngram_range in ngram_options:
            for clf_name, classifier in classifier_options:
                config_name = f"{model_version}_feat{max_features // 1000}k_ngram{ngram_range[0]}-{ngram_range[1]}_{clf_name}"
                logging.info("=" * 60)
                logging.info(f"Training: {config_name}")
                logging.info("=" * 60)
                print("\n" + "=" * 60)
                print(f"Training: {config_name}")
                print("=" * 60)

                # Create TF-IDF pipeline
                logging.info(f"Creating pipeline with max_features={max_features}, ngram_range={ngram_range}, classifier={clf_name}")
                print(f"Creating pipeline with max_features={max_features}, ngram_range={ngram_range}, classifier={clf_name}")
                text_clf = Pipeline([
                    ('vect', CountVectorizer(max_features=max_features, ngram_range=ngram_range)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('clf', classifier)
                ])

                # Train the model
                logging.info("Training model...")
                print("Training model...")
                start_time = time.time()
                text_clf.fit(X_train_raw, y_train)
                train_time = time.time() - start_time
                logging.info(f"Training completed in {train_time:.2f} seconds")
                print(f"Training completed in {train_time:.2f} seconds")

                # Evaluate on training set
                logging.info("Evaluating on training set...")
                print("Evaluating on training set...")
                train_predictions = text_clf.predict(X_train_raw)
                train_accuracy = accuracy_score(y_train, train_predictions)
                logging.info(f"Training accuracy: {train_accuracy:.4f}")
                print(f"Training accuracy: {train_accuracy:.4f}")

                # Evaluate on test set
                logging.info("Evaluating on test set...")
                print("Evaluating on test set...")
                start_time = time.time()
                test_predictions = text_clf.predict(X_test_raw)
                test_accuracy = accuracy_score(y_test, test_predictions)
                prediction_time = time.time() - start_time
                logging.info(f"Test accuracy: {test_accuracy:.4f}")
                logging.info(f"Prediction time: {prediction_time:.2f} seconds")
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
                logging.info(f"Result stored: {config_name} - Test accuracy: {test_accuracy:.4f}")

                # Show classification report for all classes
                logging.info("Classification Report (all classes):")
                print("\nClassification Report (all classes):")
                report = classification_report(y_test, test_predictions, zero_division=0)
                logging.info(report)
                print(report)

                # Models are saved at the end to avoid clutter

    # Print summary of all experiments
    logging.info("=" * 80)
    logging.info("EXPERIMENT SUMMARY")
    logging.info("=" * 80)
    summary_header = f"{'Config':<50} {'Train Acc':<10} {'Test Acc':<10} {'Train Time':<12} {'Pred Time':<10}"
    logging.info(summary_header)
    logging.info("-" * 80)
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(summary_header)
    print("-" * 80)
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
        summary_line = f"{result['config_name']:<50} {result['train_accuracy']:.4f}     {result['test_accuracy']:.4f}      {result['train_time']:>8.2f}s    {result['prediction_time']:>6.2f}s"
        logging.info(summary_line)
        print(summary_line)

    # Save best model as the main model
    best_result = max(results, key=lambda x: x['test_accuracy'])
    best_msg = f"Best configuration: {best_result['config_name']} with test accuracy: {best_result['test_accuracy']:.4f}"
    logging.info(best_msg)
    print(f"\n{best_msg}")

    # Load and save best model as main model
    import joblib
    best_model_path = os.path.join(output_folder, f"{best_result['config_name']}.pkl")
    best_model = joblib.load(best_model_path)
    main_model_path = os.path.join(output_folder, 'model.pkl')
    joblib.dump(best_model, main_model_path)
    logging.info(f"Best model saved as main model to {main_model_path}")
    print(f"Best model saved as main model to {main_model_path}")

    # Save results to JSON
    results_file = os.path.join(output_folder, f'{model_version}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_file}")
    print(f"Results saved to {results_file}")

    # Save results to run directory
    run_results_file = os.path.join(run_dir, 'all_results.json')
    with open(run_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to run directory: {run_results_file}")

    # Save dataset info
    dataset_info_file = os.path.join(output_folder, 'dataset_info.json')
    with open(dataset_info_file, 'w') as f:
        json.dump({
            'dataset_name': dataset_name,
            'n_samples': n_samples,
            'info': info
        }, f, indent=2)
    logging.info(f"Dataset info saved to {dataset_info_file}")

    # Save run metadata
    run_metadata = {
        'timestamp': timestamp,
        'model_version': model_version,
        'dataset_name': dataset_name,
        'n_samples': n_samples,
        'total_configs': len(results),
        'best_config': best_result,
        'all_results': results,
        'dataset_info': info
    }

    metadata_file = os.path.join(run_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(run_metadata, f, indent=2)
    logging.info(f"Run metadata saved to {metadata_file}")

    # Save label mapping for reference
    label_mapping_filename = os.path.join(output_folder, 'labels.txt')
    with open(label_mapping_filename, 'w', encoding='utf-8') as f:
        for label in sorted(set(y_train)):
            f.write(f"{label}\n")
    logging.info(f"Label mapping saved to {label_mapping_filename}")
    print(f"Label mapping saved to {label_mapping_filename}")

    logging.info(f"All models training run completed: {timestamp}")

    return results


def main():
    """Main function with argument parsing"""
    available_datasets = list_datasets()
    parser = argparse.ArgumentParser(description='Train text classification models')
    parser.add_argument('--dataset', type=str, choices=available_datasets, default='vntc',
                        help=f'Dataset to use (default: vntc). Available: {available_datasets}')
    parser.add_argument('--model', type=str, choices=['svc', 'logistic'], default='svc',
                        help='Model to train (default: svc)')
    parser.add_argument('--max-features', type=int, default=20000,
                        help='Maximum number of features for TF-IDF (default: 20000)')
    parser.add_argument('--ngram-min', type=int, default=1,
                        help='Minimum n-gram range (default: 1)')
    parser.add_argument('--ngram-max', type=int, default=2,
                        help='Maximum n-gram range (default: 2)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Limit number of samples per split for quick testing (default: None)')
    parser.add_argument('--all-models', action='store_true',
                        help='Train all model combinations instead of single model')

    args = parser.parse_args()

    if args.all_models:
        print(f"Training all model combinations on {args.dataset} dataset...")
        if args.n_samples:
            print(f"Using sample limit: {args.n_samples}")
        train_all_models(dataset_name=args.dataset, n_samples=args.n_samples)
    else:
        print(f"Training single model: {args.model} on {args.dataset} dataset")
        if args.n_samples:
            print(f"Using sample limit: {args.n_samples}")
        ngram_range = (args.ngram_min, args.ngram_max)
        result = train_single_model(
            model_name=args.model,
            dataset_name=args.dataset,
            vect_max_features=args.max_features,
            ngram_range=ngram_range,
            n_samples=args.n_samples
        )
        print(f"\nTraining completed: {result['config_name']}")
        print(f"Test accuracy: {result['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
