import logging
import warnings

import joblib
from huggingface_hub import hf_hub_download

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("underthesea")

classifier = None


def _load_classifier():
    global classifier
    if not classifier:
        # Download and load UTS2017_Bank model from Hugging Face
        model_path = hf_hub_download(
            repo_id="undertheseanlp/sonar_core_1",
            filename="uts2017_bank_classifier_20250928_060819.joblib",
        )
        classifier = joblib.load(model_path)
    return classifier


def classify(X):
    classifier = _load_classifier()

    # Use predict_text function for prediction
    prediction, _, _ = predict_text(classifier, X)

    # Return as list to maintain compatibility with existing API
    return [str(prediction)]


def classify_with_confidence(X):
    classifier = _load_classifier()

    # Use predict_text function for prediction
    prediction, confidence, _ = predict_text(classifier, X)

    # Get full probabilities for backward compatibility
    probabilities = classifier.predict_proba([X])[0]

    return {"category": str(prediction), "confidence": confidence, "probabilities": probabilities}


def predict_text(model, text):
    probabilities = model.predict_proba([text])[0]

    # Get top 3 predictions sorted by probability
    top_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = []
    for idx in top_indices:
        category = model.classes_[idx]
        prob = probabilities[idx]
        top_predictions.append((category, prob))

    # The prediction should be the top category
    prediction = top_predictions[0][0]
    confidence = top_predictions[0][1]

    return prediction, confidence, top_predictions
