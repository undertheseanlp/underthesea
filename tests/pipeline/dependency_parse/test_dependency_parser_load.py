"""
Tests for DependencyParser.load() method.

This validates the fix for GitHub issue #687:
https://github.com/undertheseanlp/underthesea/issues/687

The issue was that when loading a model from a non-existent local path,
the code would incorrectly pass that path to torch.hub.load_state_dict_from_url(),
causing a confusing "ValueError: unknown url type" error.
"""
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from underthesea.models.dependency_parser import DependencyParser
from underthesea.utils.sp_init import PRETRAINED


class TestDependencyParserLoad(unittest.TestCase):
    """Tests for DependencyParser.load() error handling."""

    def test_load_nonexistent_path_raises_file_not_found_error(self):
        """
        Test that loading from a non-existent local path raises FileNotFoundError
        instead of ValueError about unknown URL type.

        This is the main fix for issue #687.
        """
        nonexistent_path = '/nonexistent/path/to/model'

        with self.assertRaises(FileNotFoundError) as context:
            DependencyParser.load(nonexistent_path)

        error_message = str(context.exception)
        self.assertIn(nonexistent_path, error_message)
        self.assertIn('Model not found', error_message)
        self.assertIn('vi-dp-v1', error_message)  # Should list available models

    def test_load_nonexistent_path_with_model_folder_pattern(self):
        """
        Test the exact scenario from issue #687: loading from ~/.underthesea/models path.
        """
        path_like_issue = '/root/.underthesea/models/parsers/vi-dp-v1a0'

        with self.assertRaises(FileNotFoundError) as context:
            DependencyParser.load(path_like_issue)

        error_message = str(context.exception)
        self.assertIn(path_like_issue, error_message)
        # Should NOT raise ValueError about unknown URL type
        self.assertNotIn('unknown url type', error_message.lower())

    def test_load_nonexistent_path_suggests_pretrained_models(self):
        """
        Test that the error message includes available pretrained model names.
        """
        with self.assertRaises(FileNotFoundError) as context:
            DependencyParser.load('/some/fake/path')

        error_message = str(context.exception)
        # Check that at least some pretrained model names are mentioned
        for model_name in ['vi-dp-v1', 'vi-dp-v1a0']:
            self.assertIn(model_name, error_message)

    @patch('underthesea.models.dependency_parser.torch.load')
    def test_load_existing_local_path_uses_torch_load(self, mock_torch_load):
        """
        Test that loading from an existing local path uses torch.load().
        """
        mock_state = {
            'args': {
                'n_words': 100,
                'n_feats': 50,
                'n_rels': 10,
                'pad_index': 0,
                'unk_index': 1,
                'feat_pad_index': 0,
            },
            'transform': MagicMock(),
            'embeddings': [],
            'pretrained': None,
            'state_dict': {},
        }
        mock_torch_load.return_value = mock_state

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        with patch.object(DependencyParser, '_init_model_with_state_dict') as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            DependencyParser.load(temp_path)

            mock_torch_load.assert_called_once()
            call_args = mock_torch_load.call_args
            self.assertEqual(call_args[0][0], temp_path)

    @patch('underthesea.models.dependency_parser.torch.hub.load_state_dict_from_url')
    def test_load_pretrained_name_uses_url_from_dict(self, mock_hub_load):
        """
        Test that loading with a pretrained model name uses the URL from PRETRAINED dict.
        """
        mock_state = {
            'args': {
                'n_words': 100,
                'n_feats': 50,
                'n_rels': 10,
                'pad_index': 0,
                'unk_index': 1,
                'feat_pad_index': 0,
            },
            'transform': MagicMock(),
            'embeddings': [],
            'pretrained': None,
            'state_dict': {},
        }
        mock_hub_load.return_value = mock_state

        with patch.object(DependencyParser, '_init_model_with_state_dict') as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            DependencyParser.load('vi-dp-v1a0')

            mock_hub_load.assert_called_once()
            call_args = mock_hub_load.call_args
            expected_url = PRETRAINED['vi-dp-v1a0']
            self.assertEqual(call_args[0][0], expected_url)

    @patch('underthesea.models.dependency_parser.torch.hub.load_state_dict_from_url')
    def test_load_http_url_uses_hub_load(self, mock_hub_load):
        """
        Test that loading with an HTTP URL uses torch.hub.load_state_dict_from_url().
        """
        mock_state = {
            'args': {
                'n_words': 100,
                'n_feats': 50,
                'n_rels': 10,
                'pad_index': 0,
                'unk_index': 1,
                'feat_pad_index': 0,
            },
            'transform': MagicMock(),
            'embeddings': [],
            'pretrained': None,
            'state_dict': {},
        }
        mock_hub_load.return_value = mock_state

        test_url = 'https://example.com/model.pt'

        with patch.object(DependencyParser, '_init_model_with_state_dict') as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            DependencyParser.load(test_url)

            mock_hub_load.assert_called_once()
            call_args = mock_hub_load.call_args
            self.assertEqual(call_args[0][0], test_url)

    @patch('underthesea.models.dependency_parser.torch.hub.load_state_dict_from_url')
    def test_load_https_url_uses_hub_load(self, mock_hub_load):
        """
        Test that loading with an HTTPS URL uses torch.hub.load_state_dict_from_url().
        """
        mock_state = {
            'args': {
                'n_words': 100,
                'n_feats': 50,
                'n_rels': 10,
                'pad_index': 0,
                'unk_index': 1,
                'feat_pad_index': 0,
            },
            'transform': MagicMock(),
            'embeddings': [],
            'pretrained': None,
            'state_dict': {},
        }
        mock_hub_load.return_value = mock_state

        test_url = 'http://example.com/model.pt'

        with patch.object(DependencyParser, '_init_model_with_state_dict') as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            DependencyParser.load(test_url)

            mock_hub_load.assert_called_once()
            call_args = mock_hub_load.call_args
            self.assertEqual(call_args[0][0], test_url)


if __name__ == '__main__':
    unittest.main()
