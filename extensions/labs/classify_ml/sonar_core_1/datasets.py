import os
import urllib.request
import zipfile
from abc import ABC, abstractmethod


class Dataset(ABC):
    """Base class for all datasets"""

    @abstractmethod
    def load_data(self):
        """Load dataset and return X, y"""
        pass

    @abstractmethod
    def get_info(self):
        """Return dataset information"""
        pass


class TextClassificationDataset(Dataset):
    """Base class for text classification datasets with common functionality"""

    def __init__(self, dataset_folder=None, n_samples=None):
        self.dataset_folder = dataset_folder
        self.n_samples = n_samples
        self.train_file = None
        self.test_file = None
        self.dataset_name = None
        self.dataset_description = None
        self.download_url = None

    def _ensure_dataset_exists(self):
        """Download dataset if not exists"""
        if not os.path.exists(self.train_file) or not os.path.exists(self.test_file):
            print(f"Dataset not found. Downloading {self.dataset_name} dataset...")

            # Create directories
            os.makedirs(os.path.dirname(self.dataset_folder), exist_ok=True)

            # Download zip file
            zip_filename = os.path.basename(self.download_url)
            zip_path = os.path.join(os.path.dirname(self.dataset_folder), zip_filename)

            print(f"Downloading from {self.download_url}...")
            urllib.request.urlretrieve(self.download_url, zip_path)

            # Extract zip file
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(self.dataset_folder))

            # Clean up zip file
            os.remove(zip_path)
            print("Dataset downloaded and extracted successfully!")

    def _parse_file(self, file_path, n_samples=None):
        """Parse a single data file with optional sample limit"""
        X_raw = []
        y = []
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n_samples is not None and i >= n_samples:
                    break
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    label = parts[0].replace("__label__", "")
                    text = parts[1]
                    y.append(label)
                    X_raw.append(text)
        return X_raw, y

    def load_data(self):
        """Load training and test data"""
        print(f"Loading {self.dataset_name} dataset...")
        if self.n_samples:
            print(f"Loading up to {self.n_samples} samples per split...")

        # Load training data
        print("Reading train.txt...")
        X_train_raw, y_train = self._parse_file(self.train_file, self.n_samples)

        # Load test data
        print("Reading test.txt...")
        X_test_raw, y_test = self._parse_file(self.test_file, self.n_samples)

        return (X_train_raw, y_train), (X_test_raw, y_test)

    def get_info(self):
        """Get dataset information"""
        (X_train_raw, y_train), (X_test_raw, y_test) = self.load_data()

        info = {
            "name": self.dataset_name,
            "description": self.dataset_description,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "unique_labels": len(set(y_train)),
            "labels": sorted(set(y_train)),
        }

        return info


class VNTCDataset(TextClassificationDataset):
    """VNTC Vietnamese Text Classification Dataset"""

    def __init__(self, dataset_folder=None, n_samples=None):
        super().__init__(dataset_folder, n_samples)

        if self.dataset_folder is None:
            self.dataset_folder = os.path.expanduser("~/.underthesea/VNTC")

        self.train_file = os.path.join(self.dataset_folder, "train.txt")
        self.test_file = os.path.join(self.dataset_folder, "test.txt")
        self.dataset_name = "VNTC"
        self.dataset_description = "Vietnamese Text Classification Dataset"
        self.download_url = (
            "https://github.com/undertheseanlp/underthesea/releases/download/resources/VNTC.zip"
        )

        # Download dataset if not exists
        self._ensure_dataset_exists()


class UTS2017BankDataset(TextClassificationDataset):
    """UTS2017 Bank Vietnamese Text Classification Dataset"""

    def __init__(self, dataset_folder=None, n_samples=None):
        super().__init__(dataset_folder, n_samples)

        if self.dataset_folder is None:
            self.dataset_folder = os.path.expanduser("~/.underthesea/UTS2017_BANK")

        self.train_file = os.path.join(self.dataset_folder, "train.txt")
        self.test_file = os.path.join(self.dataset_folder, "test.txt")
        self.dataset_name = "UTS2017_BANK"
        self.dataset_description = "UTS2017 Bank Vietnamese Text Classification Dataset"
        self.download_url = "https://github.com/undertheseanlp/underthesea/releases/download/resources/UTS2017_BANK.zip"

        # Download dataset if not exists
        self._ensure_dataset_exists()


# Dataset registry
DATASETS = {
    "vntc": VNTCDataset,
    "uts2017_bank": UTS2017BankDataset,
}


def load_dataset(name, dataset_folder=None, n_samples=None, **kwargs):
    """
    Load a dataset by name, similar to Hugging Face's load_dataset function.

    Args:
        name: Name of the dataset to load ('vntc', 'uts2017_bank')
        dataset_folder: Optional custom folder path for the dataset
        n_samples: Optional limit on number of samples to load per split (train/test)
        **kwargs: Additional arguments to pass to the dataset constructor

    Returns:
        Dataset instance that can be used to load data and get info

    Examples:
        >>> # Load full dataset
        >>> dataset = load_dataset('vntc')
        >>> (X_train, y_train), (X_test, y_test) = dataset.load_data()

        >>> # Load only 1000 samples per split for quick testing
        >>> dataset = load_dataset('vntc', n_samples=1000)
        >>> (X_train, y_train), (X_test, y_test) = dataset.load_data()

        >>> # Load UTS2017_BANK with sample limit
        >>> dataset = load_dataset('uts2017_bank', n_samples=500)
        >>> info = dataset.get_info()
    """
    name_lower = name.lower()

    if name_lower not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")

    dataset_class = DATASETS[name_lower]
    return dataset_class(dataset_folder=dataset_folder, n_samples=n_samples, **kwargs)


def list_datasets():
    """
    List all available datasets.

    Returns:
        List of available dataset names
    """
    return list(DATASETS.keys())
