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


def classify(X):
    global classifier

    if not classifier:
        # Download and load UTS2017_Bank model from Hugging Face
        model_path = hf_hub_download(
            repo_id="undertheseanlp/sonar_core_1",
            filename="uts2017_bank_classifier_20250927_161733.joblib",
        )
        classifier = joblib.load(model_path)

    # Make prediction
    prediction = classifier.predict([X])[0]

    # Return as list to maintain compatibility with existing API
    return [prediction]


def classify_with_confidence(X):
    global classifier

    if not classifier:
        # Download and load UTS2017_Bank model from Hugging Face
        model_path = hf_hub_download(
            repo_id="undertheseanlp/sonar_core_1",
            filename="uts2017_bank_classifier_20250927_161733.joblib",
        )
        classifier = joblib.load(model_path)

    # Make prediction with probabilities
    prediction = classifier.predict([X])[0]
    probabilities = classifier.predict_proba([X])[0]
    confidence = max(probabilities)

    return {"category": prediction, "confidence": confidence, "probabilities": probabilities}
