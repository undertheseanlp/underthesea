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
        # Download and load UTS2017_Sentiment model from Hugging Face
        print("Downloading Pulse Core 1 (Vietnamese Banking Aspect Sentiment) "
              "model from Hugging Face Hub...")

        try:
            model_path = hf_hub_download(
                repo_id="undertheseanlp/pulse_core_1",
                filename="uts2017_sentiment_20250928_131716.joblib",
            )
            print(f"Model downloaded to: {model_path}")

            print("Loading model...")
            classifier = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to download or load model: {e}")
            raise
    return classifier


def sentiment(X) -> list[str]:
    classifier = _load_classifier()

    # Use predict_text function for prediction
    prediction, _, _ = predict_text(classifier, X)

    # Return as list to maintain compatibility with existing API
    return [str(prediction)]


def sentiment_with_confidence(X):
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


def get_labels():
    """Get all available sentiment labels for the bank classifier

    Returns:
        list: A list of all sentiment labels that the classifier can predict
    """
    classifier = _load_classifier()
    return [str(label) for label in classifier.classes_]
