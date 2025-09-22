import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from datasets import (
    DATASETS,
    Dataset,
    TextClassificationDataset,
    UTS2017BankDataset,
    VNTCDataset,
    list_datasets,
    load_dataset,
)


class TestDatasetBase:
    """Test the abstract Dataset base class"""

    def test_dataset_is_abstract(self):
        """Test that Dataset cannot be instantiated directly"""
        with pytest.raises(TypeError):
            Dataset()

    def test_dataset_requires_methods(self):
        """Test that subclasses must implement required methods"""

        class IncompleteDataset(Dataset):
            pass

        with pytest.raises(TypeError):
            IncompleteDataset()


class TestTextClassificationDataset:
    """Test TextClassificationDataset base class"""

    def test_text_classification_dataset_is_concrete(self):
        """Test that TextClassificationDataset can be instantiated"""
        dataset = TextClassificationDataset()
        assert dataset.dataset_folder is None
        assert dataset.n_samples is None
        assert dataset.train_file is None
        assert dataset.test_file is None

    def test_text_classification_dataset_init_with_params(self):
        """Test initialization with parameters"""
        dataset = TextClassificationDataset(dataset_folder="/test/path", n_samples=100)
        assert dataset.dataset_folder == "/test/path"
        assert dataset.n_samples == 100

    def test_text_classification_dataset_inheritance(self):
        """Test that TextClassificationDataset inherits from Dataset"""
        assert issubclass(TextClassificationDataset, Dataset)

    @pytest.fixture
    def mock_dataset_files(self):
        """Create temporary dataset files for testing"""
        temp_dir = tempfile.mkdtemp()
        train_file = os.path.join(temp_dir, "train.txt")
        test_file = os.path.join(temp_dir, "test.txt")

        train_data = ["__label__cat1 Sample text 1", "__label__cat2 Sample text 2"]
        test_data = ["__label__cat1 Test text 1"]

        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(train_data))

        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n".join(test_data))

        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_parse_file_method(self, mock_dataset_files):
        """Test the _parse_file method"""
        dataset = TextClassificationDataset()
        train_file = os.path.join(mock_dataset_files, "train.txt")
        X, y = dataset._parse_file(train_file)

        assert len(X) == 2
        assert len(y) == 2
        assert y[0] == "cat1"
        assert "Sample text 1" in X[0]


class TestVNTCDataset:
    """Test VNTCDataset functionality"""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory for dataset testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_dataset_files(self, temp_dataset_dir):
        """Create mock train and test files"""
        train_file = os.path.join(temp_dataset_dir, "train.txt")
        test_file = os.path.join(temp_dataset_dir, "test.txt")

        # Create sample data
        train_data = [
            "__label__category1 This is a training sample 1",
            "__label__category2 This is a training sample 2",
            "__label__category1 This is a training sample 3",
        ]
        test_data = [
            "__label__category1 This is a test sample 1",
            "__label__category2 This is a test sample 2",
        ]

        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(train_data))

        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n".join(test_data))

        return temp_dataset_dir

    def test_inheritance(self):
        """Test that VNTCDataset inherits from TextClassificationDataset"""
        assert issubclass(VNTCDataset, TextClassificationDataset)
        assert issubclass(VNTCDataset, Dataset)

    def test_init_with_existing_files(self, mock_dataset_files):
        """Test initialization when dataset files already exist"""
        dataset = VNTCDataset(dataset_folder=mock_dataset_files)
        assert dataset.dataset_folder == mock_dataset_files
        assert dataset.train_file == os.path.join(mock_dataset_files, "train.txt")
        assert dataset.test_file == os.path.join(mock_dataset_files, "test.txt")
        assert dataset.n_samples is None
        assert dataset.dataset_name == "VNTC"
        assert dataset.dataset_description == "Vietnamese Text Classification Dataset"

    def test_init_with_n_samples(self, mock_dataset_files):
        """Test initialization with n_samples parameter"""
        dataset = VNTCDataset(dataset_folder=mock_dataset_files, n_samples=10)
        assert dataset.n_samples == 10

    def test_parse_file(self, mock_dataset_files):
        """Test file parsing functionality"""
        dataset = VNTCDataset(dataset_folder=mock_dataset_files)
        X, y = dataset._parse_file(dataset.train_file)

        assert len(X) == 3
        assert len(y) == 3
        assert y[0] == "category1"
        assert y[1] == "category2"
        assert "training sample 1" in X[0]

    def test_parse_file_with_n_samples(self, mock_dataset_files):
        """Test file parsing with sample limit"""
        dataset = VNTCDataset(dataset_folder=mock_dataset_files)
        X, y = dataset._parse_file(dataset.train_file, n_samples=2)

        assert len(X) == 2
        assert len(y) == 2

    def test_load_data(self, mock_dataset_files):
        """Test loading both train and test data"""
        dataset = VNTCDataset(dataset_folder=mock_dataset_files)
        (X_train, y_train), (X_test, y_test) = dataset.load_data()

        assert len(X_train) == 3
        assert len(y_train) == 3
        assert len(X_test) == 2
        assert len(y_test) == 2

    def test_load_data_with_n_samples(self, mock_dataset_files):
        """Test loading data with sample limit"""
        dataset = VNTCDataset(dataset_folder=mock_dataset_files, n_samples=1)
        (X_train, y_train), (X_test, y_test) = dataset.load_data()

        assert len(X_train) == 1
        assert len(y_train) == 1
        assert len(X_test) == 1
        assert len(y_test) == 1

    def test_get_info(self, mock_dataset_files):
        """Test getting dataset information"""
        dataset = VNTCDataset(dataset_folder=mock_dataset_files)
        info = dataset.get_info()

        assert info["name"] == "VNTC"
        assert "Vietnamese Text Classification" in info["description"]
        assert info["train_samples"] == 3
        assert info["test_samples"] == 2
        assert info["unique_labels"] == 2
        assert set(info["labels"]) == {"category1", "category2"}

    @patch("os.remove")
    @patch("urllib.request.urlretrieve")
    @patch("zipfile.ZipFile")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_download_dataset(
        self, mock_makedirs, mock_exists, mock_zipfile, mock_urlretrieve, mock_remove
    ):
        """Test dataset download when files don't exist"""
        mock_exists.return_value = False
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        with patch("builtins.print"):
            VNTCDataset()

        mock_makedirs.assert_called_once()
        mock_urlretrieve.assert_called_once()
        assert "VNTC.zip" in mock_urlretrieve.call_args[0][1]
        mock_zip_instance.extractall.assert_called_once()
        mock_remove.assert_called_once()


class TestUTS2017BankDataset:
    """Test UTS2017BankDataset functionality"""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory for dataset testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_dataset_files(self, temp_dataset_dir):
        """Create mock train and test files"""
        train_file = os.path.join(temp_dataset_dir, "train.txt")
        test_file = os.path.join(temp_dataset_dir, "test.txt")

        # Create sample data
        train_data = [
            "__label__bank1 Banking sample 1",
            "__label__bank2 Banking sample 2",
        ]
        test_data = [
            "__label__bank1 Test banking sample",
        ]

        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(train_data))

        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n".join(test_data))

        return temp_dataset_dir

    def test_inheritance(self):
        """Test that UTS2017BankDataset inherits from TextClassificationDataset"""
        assert issubclass(UTS2017BankDataset, TextClassificationDataset)
        assert issubclass(UTS2017BankDataset, Dataset)

    def test_init_with_existing_files(self, mock_dataset_files):
        """Test initialization when dataset files already exist"""
        dataset = UTS2017BankDataset(dataset_folder=mock_dataset_files)
        assert dataset.dataset_folder == mock_dataset_files
        assert dataset.n_samples is None
        assert dataset.dataset_name == "UTS2017_BANK"
        assert "Bank" in dataset.dataset_description

    def test_load_data(self, mock_dataset_files):
        """Test loading data"""
        dataset = UTS2017BankDataset(dataset_folder=mock_dataset_files)
        (X_train, y_train), (X_test, y_test) = dataset.load_data()

        assert len(X_train) == 2
        assert len(y_train) == 2
        assert len(X_test) == 1
        assert len(y_test) == 1

    def test_get_info(self, mock_dataset_files):
        """Test getting dataset information"""
        dataset = UTS2017BankDataset(dataset_folder=mock_dataset_files)
        info = dataset.get_info()

        assert info["name"] == "UTS2017_BANK"
        assert "Bank" in info["description"]
        assert info["train_samples"] == 2
        assert info["test_samples"] == 1

    @patch("os.remove")
    @patch("urllib.request.urlretrieve")
    @patch("zipfile.ZipFile")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_download_dataset(
        self, mock_makedirs, mock_exists, mock_zipfile, mock_urlretrieve, mock_remove
    ):
        """Test dataset download with correct URL"""
        mock_exists.return_value = False
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        with patch("builtins.print"):
            UTS2017BankDataset()

        mock_makedirs.assert_called_once()
        mock_urlretrieve.assert_called_once()
        assert "UTS2017_BANK.zip" in mock_urlretrieve.call_args[0][0]
        mock_remove.assert_called_once()


class TestLoadDataset:
    """Test the load_dataset function"""

    @patch("os.path.exists")
    def test_load_dataset_vntc(self, mock_exists):
        """Test loading VNTC dataset by name"""
        mock_exists.return_value = True
        with patch.object(VNTCDataset, "_ensure_dataset_exists"):
            dataset = load_dataset("vntc")
            assert isinstance(dataset, VNTCDataset)

    @patch("os.path.exists")
    def test_load_dataset_uts2017_bank(self, mock_exists):
        """Test loading UTS2017_BANK dataset by name"""
        mock_exists.return_value = True
        with patch.object(UTS2017BankDataset, "_ensure_dataset_exists"):
            dataset = load_dataset("uts2017_bank")
            assert isinstance(dataset, UTS2017BankDataset)

    @patch("os.path.exists")
    def test_load_dataset_with_n_samples(self, mock_exists):
        """Test loading dataset with n_samples parameter"""
        mock_exists.return_value = True
        with patch.object(VNTCDataset, "_ensure_dataset_exists"):
            dataset = load_dataset("vntc", n_samples=100)
            assert isinstance(dataset, VNTCDataset)
            assert dataset.n_samples == 100

    @patch("os.path.exists")
    def test_load_dataset_with_custom_folder(self, mock_exists):
        """Test loading dataset with custom folder"""
        mock_exists.return_value = True
        custom_path = "/tmp/custom_path"
        with patch.object(VNTCDataset, "_ensure_dataset_exists"):
            dataset = load_dataset("vntc", dataset_folder=custom_path)
            assert isinstance(dataset, VNTCDataset)
            assert dataset.dataset_folder == custom_path

    @patch("os.path.exists")
    def test_load_dataset_case_insensitive(self, mock_exists):
        """Test that dataset names are case insensitive"""
        mock_exists.return_value = True
        with patch.object(VNTCDataset, "_ensure_dataset_exists"):
            dataset = load_dataset("VNTC")
            assert isinstance(dataset, VNTCDataset)

    def test_load_dataset_invalid_name(self):
        """Test error handling for invalid dataset name"""
        with pytest.raises(ValueError) as exc_info:
            load_dataset("invalid_dataset")

        assert "Dataset 'invalid_dataset' not found" in str(exc_info.value)
        assert "vntc" in str(exc_info.value).lower()
        assert "uts2017_bank" in str(exc_info.value).lower()


class TestListDatasets:
    """Test the list_datasets function"""

    def test_list_datasets(self):
        """Test that list_datasets returns all available datasets"""
        datasets = list_datasets()
        assert "vntc" in datasets
        assert "uts2017_bank" in datasets
        assert len(datasets) == len(DATASETS)

    def test_list_datasets_returns_list(self):
        """Test that list_datasets returns a list type"""
        datasets = list_datasets()
        assert isinstance(datasets, list)


class TestDatasetRegistry:
    """Test the dataset registry"""

    def test_registry_contains_all_datasets(self):
        """Test that the registry contains expected datasets"""
        assert "vntc" in DATASETS
        assert "uts2017_bank" in DATASETS
        assert DATASETS["vntc"] == VNTCDataset
        assert DATASETS["uts2017_bank"] == UTS2017BankDataset

    def test_registry_values_are_classes(self):
        """Test that registry values are dataset classes"""
        for dataset_class in DATASETS.values():
            assert issubclass(dataset_class, Dataset)
            assert issubclass(dataset_class, TextClassificationDataset)
