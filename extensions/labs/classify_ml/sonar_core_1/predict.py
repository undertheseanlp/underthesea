import glob
import json
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


def load_dataset_info(model_dir=None):
    """Load dataset information if available"""
    output_folder = get_model_folder(model_dir)
    if output_folder is None:
        return None

    dataset_info_path = os.path.join(output_folder, 'dataset_info.json')
    try:
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        return dataset_info
    except FileNotFoundError:
        return None


def load_run_metadata(model_dir=None):
    """Load run metadata if available"""
    if model_dir:
        # If specific model_dir is provided, try to find metadata in parent directory
        run_dir = os.path.dirname(model_dir) if model_dir.endswith('models') else model_dir
    else:
        run_dir = get_latest_run_dir()

    if run_dir is None:
        return None

    metadata_path = os.path.join(run_dir, 'metadata.json')
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        return None


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


def get_example_texts_for_dataset(dataset_name):
    """Get appropriate example texts based on the dataset used"""
    if dataset_name == 'vntc':
        return [
            "Việt Nam giành chiến thắng 3-0 trước Thái Lan trong trận bán kết",
            "Apple ra mắt iPhone mới với nhiều tính năng đột phá",
            "Cách nấu phở bò ngon đúng chuẩn Hà Nội",
            "Phát hiện vaccine mới chống lại virus corona",
            "Thị trường chứng khoán tăng điểm mạnh trong phiên sáng nay",
            "Thời tiết hôm nay có mưa rào và dông vào chiều tối",
            "Nhà khoa học Việt Nam đạt giải thưởng quốc tế",
            "Đội tuyển bóng đá nữ Việt Nam vào chung kết"
        ]
    elif dataset_name == 'uts2017_bank':
        return [
            "Tôi muốn gửi tiết kiệm với lãi suất cao nhất",
            "Làm thế nào để vay tiền mua nhà với lãi suất ưu đãi?",
            "Tôi cần mở tài khoản thanh toán mới",
            "Phí chuyển khoản quốc tế là bao nhiêu?",
            "Tôi muốn đăng ký thẻ tín dụng với hạn mức cao",
            "Cách tính lãi suất khi gửi tiết kiệm có kỳ hạn",
            "Tôi cần hỗ trợ về dịch vụ internet banking",
            "Làm sao để khóa thẻ ATM khi bị mất?"
        ]
    else:
        # Default examples for unknown datasets
        return [
            "Đây là một văn bản mẫu để kiểm tra",
            "Hệ thống phân loại văn bản tiếng Việt",
            "Kiểm tra khả năng dự đoán của mô hình"
        ]


def interactive_predict(model, dataset_name=None):
    """Interactive mode for predictions"""
    dataset_info = f" ({dataset_name} dataset)" if dataset_name else ""
    print(f"\nInteractive prediction mode{dataset_info} (type 'quit' to exit)")
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


def show_model_info(model_dir=None):
    """Show information about the loaded model"""
    dataset_info = load_dataset_info(model_dir)
    metadata = load_run_metadata(model_dir)

    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    if metadata:
        print(f"Training timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"Model configuration: {metadata.get('config_name', 'Unknown')}")
        print(f"Dataset used: {metadata.get('dataset_name', 'Unknown')}")
        if metadata.get('n_samples'):
            print(f"Sample limit: {metadata['n_samples']}")
        print(f"Test accuracy: {metadata.get('test_accuracy', 'Unknown'):.4f}")
        print(f"Training time: {metadata.get('train_time', 'Unknown'):.2f}s")

    if dataset_info:
        info = dataset_info.get('info', {})
        print(f"Training samples: {info.get('train_samples', 'Unknown')}")
        print(f"Test samples: {info.get('test_samples', 'Unknown')}")
        print(f"Unique labels: {info.get('unique_labels', 'Unknown')}")

    return metadata.get('dataset_name') if metadata else None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Vietnamese Text Classification Prediction')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--examples', action='store_true', help='Show example predictions')
    parser.add_argument('--info', action='store_true', help='Show model information')
    parser.add_argument('--model-dir', type=str, help='Specific model directory to load from (default: latest run)')
    args = parser.parse_args()

    # Load the model and labels
    model = load_model(model_dir=args.model_dir)
    labels = load_labels(model_dir=args.model_dir)

    # Show model information
    dataset_name = show_model_info(model_dir=args.model_dir)

    print(f"\nAvailable categories ({len(labels)}): {', '.join(labels[:10])}")
    if len(labels) > 10:
        print(f"... and {len(labels) - 10} more categories")

    if args.info:
        # Just show info and exit
        return

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
        example_texts = get_example_texts_for_dataset(dataset_name)

        print("\n" + "=" * 60)
        print("EXAMPLE PREDICTIONS")
        print("=" * 60)

        results = predict_batch(model, example_texts)
        for result in results:
            print(f"\nText: {result['text']}")
            print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")

    if args.interactive:
        # Interactive mode
        interactive_predict(model, dataset_name)


if __name__ == "__main__":
    main()
