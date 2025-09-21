import os
import sys
import urllib.request
import zipfile
from os.path import dirname

import joblib

sys.path.insert(0, dirname(dirname(__file__)))
classifier = None


def _ensure_model_exists():
    """Get model from latest run or download from release"""
    model_dir = os.path.expanduser("~/.underthesea/models")
    model_file = os.path.join(model_dir, "sonar_core_1.pkl")
    labels_file = os.path.join(model_dir, "sonar_core_1_labels.txt")

    # Check if model already exists
    if os.path.exists(model_file) and os.path.exists(labels_file):
        return model_file, labels_file

    # Try to get from latest local run first
    runs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "extensions", "labs", "classify_ml", "sonar_core_1", "runs")
    if os.path.exists(runs_dir):
        import glob
        run_dirs = glob.glob(os.path.join(runs_dir, "[0-9]*_[0-9]*"))
        if run_dirs:
            latest_run = sorted(run_dirs)[-1]
            latest_model = os.path.join(latest_run, "models", "model.pkl")
            latest_labels = os.path.join(latest_run, "models", "labels.txt")

            if os.path.exists(latest_model) and os.path.exists(latest_labels):
                print(f"Using model from latest local run: {latest_run}")
                os.makedirs(model_dir, exist_ok=True)

                # Copy from latest run
                import shutil
                shutil.copy2(latest_model, model_file)
                shutil.copy2(latest_labels, labels_file)
                return model_file, labels_file

    print("Downloading Sonar Core 1 model...")

    # Create directories
    os.makedirs(model_dir, exist_ok=True)

    # Download zip file
    zip_url = "https://github.com/undertheseanlp/underthesea/releases/download/resources/sonar_core_1.zip"
    zip_path = os.path.join(model_dir, "sonar_core_1.zip")

    print(f"Downloading from {zip_url}...")
    urllib.request.urlretrieve(zip_url, zip_path)

    # Extract zip file
    print("Extracting model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    # Clean up zip file
    os.remove(zip_path)

    # Rename extracted files to expected names
    extracted_model = os.path.join(model_dir, "model.pkl")
    extracted_labels = os.path.join(model_dir, "labels.txt")

    if os.path.exists(extracted_model):
        os.rename(extracted_model, model_file)
    if os.path.exists(extracted_labels):
        os.rename(extracted_labels, labels_file)

    print("Sonar Core 1 model downloaded and extracted successfully!")
    return model_file, labels_file


def _load_labels(labels_file):
    """Load label mapping from file"""
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def classify(text):
    """Classify Vietnamese text using Sonar Core 1 model

    Args:
        text (str): Vietnamese text to classify

    Returns:
        list: List containing the predicted category (for compatibility with underthesea API)
    """
    global classifier

    if not classifier:
        model_file, labels_file = _ensure_model_exists()
        classifier = joblib.load(model_file)
        classifier.labels = _load_labels(labels_file)

    # Make prediction
    prediction = classifier.predict([text])[0]
    return [prediction]


def classify_with_confidence(text):
    """Classify Vietnamese text with confidence scores

    Args:
        text (str): Vietnamese text to classify

    Returns:
        dict: Dictionary with prediction and top 3 probabilities
    """
    global classifier

    if not classifier:
        model_file, labels_file = _ensure_model_exists()
        classifier = joblib.load(model_file)
        classifier.labels = _load_labels(labels_file)

    # Make prediction with probabilities
    prediction = classifier.predict([text])[0]
    probabilities = classifier.predict_proba([text])[0]

    # Get top 3 predictions with probabilities
    classes = classifier.classes_
    prob_dict = dict(zip(classes, probabilities))
    top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        'prediction': prediction,
        'confidence': top_predictions[0][1],
        'top_3': top_predictions
    }
