import glob
import os
import sys

import joblib


def get_latest_run_dir():
    """Get the latest run directory based on timestamp"""
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return None

    run_dirs = glob.glob(os.path.join(runs_dir, "[0-9]*_[0-9]*"))
    if not run_dirs:
        return None

    # Sort by timestamp (directory name) and get the latest
    latest_run = sorted(run_dirs)[-1]
    return latest_run


def get_model_folder(model_dir=None):
    """Get model folder, either from specific model_dir or latest run"""
    if model_dir:
        if os.path.exists(model_dir):
            return model_dir
        else:
            print(f"Error: Model directory '{model_dir}' not found.")
            return None
    else:
        latest_run = get_latest_run_dir()
        if latest_run:
            return os.path.join(latest_run, "models")
        else:
            print("Error: No run directories found.")
            return None


def load_model(model_path=None, model_dir=None):
    """Load the trained model from disk"""
    if model_path is None:
        output_folder = get_model_folder(model_dir)
        if output_folder is None:
            sys.exit(1)
        model_path = os.path.join(output_folder, 'model.pkl')
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(
            f"Error: Model file '{model_path}' not found. "
            "Please run train.py first to train the model."
        )
        sys.exit(1)


def load_labels(label_path=None, model_dir=None):
    """Load the label mapping from disk"""
    if label_path is None:
        output_folder = get_model_folder(model_dir)
        if output_folder is None:
            sys.exit(1)
        label_path = os.path.join(output_folder, 'labels.txt')
    try:
        with open(label_path, encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except FileNotFoundError:
        print(
            f"Error: Label file '{label_path}' not found. "
            "Please run train.py first to train the model."
        )
        sys.exit(1)


def predict_text(model, text):
    """Predict the category of a single text"""
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    # Get top 3 predictions with probabilities
    classes = model.classes_
    prob_dict = dict(zip(classes, probabilities))
    top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    return prediction, top_predictions


def predict_batch(model, texts):
    """Predict categories for multiple texts"""
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results = []
    for i, text in enumerate(texts):
        prob_dict = dict(zip(model.classes_, probabilities[i]))
        top_preds = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': predictions[i],
            'confidence': top_preds[0][1],
            'top_3': top_preds
        })

    return results


def interactive_predict(model):
    """Interactive mode for predictions"""
    print("\nInteractive prediction mode (type 'quit' to exit)")
    print("-" * 50)

    while True:
        text = input("\nEnter Vietnamese text to classify: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break

        if not text:
            print("Please enter some text.")
            continue

        prediction, top_predictions = predict_text(model, text)

        print(f"\nPredicted category: {prediction}")
        print("\nTop 3 predictions with confidence:")
        for i, (label, prob) in enumerate(top_predictions, 1):
            print(f"  {i}. {label}: {prob:.2%}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Vietnamese Text Classification')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--examples', action='store_true', help='Show example predictions')
    parser.add_argument('--model-dir', type=str, help='Specific model directory to load from (default: latest run)')
    args = parser.parse_args()

    # Load the model and labels
    model = load_model(model_dir=args.model_dir)
    labels = load_labels(model_dir=args.model_dir)

    print(f"\nAvailable categories: {', '.join(labels)}")

    if args.text:
        # Single text prediction
        prediction, top_predictions = predict_text(model, args.text)
        print(f"\nText: {args.text}")
        print(f"Predicted category: {prediction}")
        print("\nTop 3 predictions with confidence:")
        for i, (label, prob) in enumerate(top_predictions, 1):
            print(f"  {i}. {label}: {prob:.2%}")

    elif args.examples or (not args.interactive and not args.text):
        # Example predictions
        example_texts = [
            "Việt Nam giành chiến thắng 3-0 trước Thái Lan trong trận bán kết",
            "Apple ra mắt iPhone mới với nhiều tính năng đột phá",
            "Cách nấu phở bò ngon đúng chuẩn Hà Nội",
            "Phát hiện vaccine mới chống lại virus corona",
            "Thị trường chứng khoán tăng điểm mạnh trong phiên sáng nay"
        ]

        print("\n" + "=" * 60)
        print("EXAMPLE PREDICTIONS")
        print("=" * 60)

        results = predict_batch(model, example_texts)
        for result in results:
            print(f"\nText: {result['text']}")
            print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")

    if args.interactive:
        # Interactive mode
        interactive_predict(model)


if __name__ == "__main__":
    main()
